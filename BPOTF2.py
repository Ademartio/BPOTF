import numpy as np
from ldpc import bp_decoder, bposd_decoder
import copy
import time
import stim
from beliefmatching import detector_error_model_to_check_matrices
from scipy.io import savemat
import scipy.io as sio
conts = sio.loadmat('transferMatrixTon.mat')

# This class will attempt to correct BB codes under circuit-level noise
class UFCLN:
    def __init__(self,
                 dem : stim.DetectorErrorModel,
                 allow_undecomposed_hyperedges: bool = True
                 ):
        
        bm = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges)
        
        self.H = bm.check_matrix.toarray()
        self.obs= bm.observables_matrix.toarray()
        self.priors= bm.priors
        
        columns_to_consider = np.zeros(self.H.shape[1])
        
        for column in range(self.H.shape[1]):
            if sum(self.H[:,column]) <= 3:
                columns_to_consider[column] = 1
        
        self.H_phen = self.H[:,np.where(columns_to_consider==1)[0]]
        self.obs_phen = self.obs[:,np.where(columns_to_consider==1)[0]]
        
        # Post processing decoding:
        
        # p = 0.001
        # bposd = bposd_decoder(
        #     self.H_phen,
        #     error_rate = p,
        #     max_iter = 20,
        #     osd_method = "osd_0"
        # )
        # self.transf_M = np.zeros((self.H_phen.shape[1],self.H.shape[1]))
        
        # for column in range(self.H.shape[1]):
        #     # print(f'Número columna en HCLN: {column}')
        #     syndrome = self.H[:,column]
        #     column_to_add = bposd.decode(syndrome)
        #     # print(f'Peso columna: {np.sum(column_to_add)}\n')
        #     # TODO quizás siguiente linea esté mal
        #     self.transf_M[:,column] = column_to_add
        self.transf_M = conts['transfMat']
        self.priors_phen = self.transf_M @ self.priors
        self.rows, self.columns = self.H_phen.shape
        savemat("gevolvetest.mat", {'H': self.H, 'obs' : self.obs,'H_phen' : self.H_phen, 'obs_phen' : self.obs_phen})
        
        
        self._bpd = bp_decoder(
            self.H,
            channel_probs = self.priors,
            max_iter = 100
        )
        # Para hacer el segundo BP, en la matriz de los edges y, futúramente fenomenológica
        self._bpd2 = bp_decoder(
            self.H_phen,
            channel_probs = self.transf_M @ self.priors,
            max_iter = 100
        )

        self._bpd3 = bp_decoder(
            self.H_phen,
            channel_probs = self.transf_M @ self.priors,
            max_iter = 100
        )
        
        # Definimos el número máximo de índices por columna
        max_nontrivial_per_col = np.max(np.sum(self.H_phen == 1, axis=0))
        # Definimos la matriz de índices. Cuando un valor es -1, es que ya no está incluido.
        self.index_matrix = np.full((max_nontrivial_per_col, self.columns), -1, dtype=np.int64)
        
        for column in range(self.columns):
            # Get the row indices where the value is 1 in the current column
            row_indices = np.where(self.H_phen[:, column] == 1)[0]
            # Place these indices in the corresponding column of index_matrix
            self.index_matrix[:len(row_indices), column] = row_indices
        
        # print('Preprocessing ready!')
        
    def SetCluster(self):
        """Set the clusters which will be grown via Union Find.

        Returns:
            cluster: array of duples. First element of the duple is the pointer to the root of the tree.
                second element is the weight of the tree.
            
        """
        cluster =  np.zeros((self.rows, 2), dtype=int)
        cluster[:, 0] = np.arange(self.rows)
        return cluster
    
    
    def sort_matrix(self, llrs):
        sorted_indices = np.argsort(llrs)[::-1]
        return  sorted_indices
    
    def Find(self, x :int, cluster_array = np.array):
        """Searches the root of a specific node on the cluster.
        
        x is the location of the leaf node.
        cluster array is the cluster

        Returns:
            cluster: array of duples. First element of the duple is the pointer to the root of the tree.
                second element is the weight of the tree.
            
        """
        while x != cluster_array[x,0]:
            x = cluster_array[x,0]
        return x, cluster_array[x,1]
    
    # def Union(self, elements:list):
    #     pass
    
    def Kruskal_hypergraph(self, sorted_indices: np.ndarray):
        # Empezamos el cluster
        cluster_array = self.SetCluster()
        # La siguiente columna  indica qué columnas nos quedaremos como árbol.
        columns_chosen = np.zeros(self.columns, dtype = bool)
        # Iteramos sobre todas las columnas de la matriz self.Hog
        counter = 0
        for column in range(self.columns):
            # Checker nos sirve para ever que checks coge cada evento.
            checker = np.zeros(self.rows, dtype = bool)
            # Depths: (root, depth, boolean about increasing tree size)
            depths = [0, -1, False]
            boolean_condition = True
            # Miramos para la columna "sorted_indices[column]]" de la matriz "self.Hog", que filas son no triviales.
            non_trivial_elements_in_column = self.index_matrix[:,sorted_indices[column]]
            # Las siguientes lineas son el método Union Find que hablamos el otro día:
            for non_trivial_element in non_trivial_elements_in_column:
                if non_trivial_element == -1:
                    break
                # Miramos la raiz y la profundidad del árbol al que corresponden.
                root, depth = self.Find(non_trivial_element, cluster_array)
                # Si llega a -1 es que no quedan checks.
                # Si se cumple esta condición quere decir que accede dos veces al mismo root.
                if checker[root]:
                    boolean_condition = False
                    break
                # Si no lo apuntamos en el checker.
                checker[root] = True
                # Mantenemos el root de profundida más grande.
                if depth > depths[1]:
                    depths = [root, depth, False]
                # Si hay más de un árbol de profundidad máxima, la profundidad del árbol aumenta.
                if depth == depths[1]:
                    depths[2] = True
            # Si la hyperarista no ha incidido más de una vez a ningún árbol la añadimos.
            if boolean_condition:
                non_trivial_checker = np.where(checker == 1)[0]
                for element in non_trivial_checker:
                    cluster_array[element,0] = depths[0]
                if  depths[2]:
                    cluster_array[depths[0]][1] += 1
                columns_chosen[sorted_indices[column]] = True
                # counter += 1
                # if counter == self.n_cols:
                #     break
        # print(f'Number of columns {counter}')
        # La siguiente lista nos indica qué columnas han sido elegidas.
        indices_columns_chosen = np.where(columns_chosen==True)[0]
        # print(f'Numero de columnas a considerar {len(indices_columns_chosen)}')
        # returned_H = H_sorted[:-2,indices_columns_chosen]
        # new_rows, new_cols = returned_H.shape
        # if new_rows >= new_cols:
        #     returned_H = np.hstack((returned_H,np.zeros((new_cols, new_rows-new_cols+1))))
        return indices_columns_chosen
                
    def decode(self, syndrome : np.array):
        # Usamos el primer BP para encontrar el error.
        recovered_error = self._bpd.decode(syndrome)
        # Si converge devolvemos el error.
        if self._bpd.converge:
            recovered_error = (self.obs @ recovered_error) % 2
            return recovered_error, 0
        print('BP2')
        # Si no converge, nos quedamos con las llrs.
        llrs = self._bpd.log_prob_ratios
        ps_h = 1 / (1 + np.exp(llrs))
        eps = 1e-14
        ps_e = self.transf_M @ ps_h        
        ps_e[ps_e > 1 - eps] = 1 - eps
        ps_e[ps_e < eps] = eps
        
        
        # updated_probs[columns_chosen] = self.priors_phen[columns_chosen].flatten()
        self._bpd2.update_channel_probs(ps_e)
        # Luego le damos a decode.
        second_recovered_error = self._bpd2.decode(syndrome)
        
        if not self._bpd2.converge:
            # print('NO CONVERGENCE')
            # return np.zeros(self.obs.shape[0]), 0
            print('OTF \n')
            llrs2 = self._bpd2.log_prob_ratios
            ps_e = 1 / (1 + np.exp(llrs2))
            eps = 1e-14      
            ps_e[ps_e > 1 - eps] = 1 - eps
            ps_e[ps_e < eps] = eps
            
            sorted_indices = self.sort_matrix(ps_e)
            a = time.perf_counter()
            # Nos quedamos con las columnas linearmente independientes.
            columns_chosen = self.Kruskal_hypergraph(sorted_indices)
            b = time.perf_counter()
            average_time = b-a
            
            updated_probs = np.full(self.columns, 1e-9)
            updated_probs[columns_chosen] = ps_e[columns_chosen]
            # updated_probs[columns_chosen] = self.priors_phen[columns_chosen]
        
        
            # updated_probs[columns_chosen] = self.priors_phen[columns_chosen].flatten()
            self._bpd3.update_channel_probs(updated_probs)
            second_recovered_error = self._bpd3.decode(syndrome)
            if not self._bpd3.converge:
                print('Rare error')
            second_recovered_error = (self.obs_phen @ second_recovered_error) % 2
            return second_recovered_error, average_time
        # else: print('Yes convergence')
        second_recovered_error = (self.obs_phen @ second_recovered_error) % 2
        # if not np.all(error_edge == second_recovered_error):
        #     pass
        return second_recovered_error, 0
    
    
    
if __name__ == "__main__":
    pass