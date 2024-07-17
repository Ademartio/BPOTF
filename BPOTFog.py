import numpy as np
from ldpc import bp_decoder
import copy
import time
import stim
from beliefmatching import detector_error_model_to_check_matrices

class UFCLN:
    
    
    def __init__(self,
                 dem: stim.DetectorErrorModel,
                 distance: int
                 ):
        
        self._model = detector_error_model_to_check_matrices(dem)
        self.H = self._model.check_matrix
        self.Hedge = self._model.edge_check_matrix
        rank = np.linalg.matrix_rank(self.Hedge.toarray())
        # print(f'Rank of H edge {rank}')
        self.priors = self._model.priors
        self.priors_edges = self._model.hyperedge_to_edge_matrix @ self.priors
        
        # Para hacer el primer BP, en el cln 
        self._bpd = bp_decoder(
            self.H,
            channel_probs = self.priors
        )
        # Para hacer el segundo BP, en la matriz de los edges y, futúramente fenomenológica
        self._bpd2 = bp_decoder(
            self.Hedge,
            channel_probs = self.priors_edges
        )

        # self.rank = np.linalg.matrix_rank(self.H)
        self.Hog = copy.deepcopy(self.Hedge).toarray().astype(np.uint8)
        self.columns = self.Hog.shape[1]
        zeros_rows = np.zeros((distance+1, self.columns))
        self.Hog =  np.vstack((self.Hog, zeros_rows))
        self.rows = self.Hog.shape[0]
        # How many weight 1 checks per measurement round
        a = []
        for column in range(self.Hog.shape[1]):
            b = (len(np.where(self.Hog[:,column] == 1)[0]))
            if b == 1:
                a.append(column)
        rate = len(a)//(distance+1)
        
        # We add virtual checks so all columns have at least two non-trivial elements.
        counter = 0
        row_to_consider = 0
        for i in range(self.columns):
            ones_in_col = np.where(self.Hog[:,i] == 1)[0]
            if len(ones_in_col) == 1:
                self.Hog[-1-row_to_consider,i] = 1
                counter += 1
                if counter == rate:
                    counter = 0
                    row_to_consider += 1
        
        # for i in range(self.columns):
        #     ones_in_col = np.where(self.Hog[:,i] == 1)[0]
        #     if len(ones_in_col) == 1:
        #         self.Hog[-1,i] = 1
        
        # Definimos el número máximo de índices por columna
        max_nontrivial_per_col = np.max(np.sum(self.Hog == 1, axis=0))
        # Definimos la matriz de índices. Cuando un valor es -1, es que ya no está incluido.
        self.index_matrix = np.full((max_nontrivial_per_col, self.columns), -1, dtype=np.int64)
        
        for column in range(self.columns):
            # Get the row indices where the value is 1 in the current column
            row_indices = np.where(self.Hog[:, column] == 1)[0]
            # Place these indices in the corresponding column of index_matrix
            self.index_matrix[:len(row_indices), column] = row_indices
        
        
    def SetCluster(self):
        """Set the clusters which will be grown via Union Find.

        Returns:
            cluster: array of duples. First element of the duple is the pointer to the root of the tree.
                second element is the weight of the tree.
            
        """
        cluster =  np.zeros((self.rows+2, 2), dtype=int)
        cluster[:, 0] = np.arange(self.rows+2)
        return cluster
    
    
    def sort_matrix(self, llrs):
        sorted_indices = np.argsort(llrs)[::-1]
        # H_sorted = self.Hog[:,sorted_indices]
        # return H_sorted, sorted_indices
        return  sorted_indices
    
    def Find(self, x :int, cluster_array = np.array):
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
                # Si otro valor no trivial te lleva a ese árbol omitimos la columna.
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
                counter += 1
                # if counter == self.n_cols:
                #     break
        # print(f'Number of columns {counter}')
        # La siguiente lista nos indica qué columnas han sido elegidas.
        indices_columns_chosen = np.where(columns_chosen==True)[0]
        # print(f'Number of considered columns OG: {len(indices_columns_chosen)}')
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
            recovered_error = (self._model.observables_matrix @ recovered_error) % 2
            return recovered_error, 0
        # Si no converge, nos quedamos con las llrs.
        llrs = self._bpd.log_prob_ratios
        ps_h = 1 / (1 + np.exp(llrs))
        eps = 1e-14
        ps_e = self._model.hyperedge_to_edge_matrix @ ps_h        
        ps_e[ps_e > 1 - eps] = 1 - eps
        ps_e[ps_e < eps] = eps
        # Ordenamos las posterioris del fenomenologico y nos quedamos con los indices por orden
        sorted_indices = self.sort_matrix(ps_e)
        a = time.perf_counter()
        # Nos quedamos con las columnas linearmente independientes.
        columns_chosen = self.Kruskal_hypergraph(sorted_indices)
        b = time.perf_counter()
        average_time = b-a
        # Para no tener que iniciar de nuevo una clase bp_decoder, usamos self._bp2.
        # Para que tenga en cuenta solo las columnas que nos interesan, lo iniciamos, le damos probabilidad 0 a las columnas que no usamos 
        # y probabilidad self.p a las que sí.
        updated_probs = np.zeros(self.columns)

        updated_probs[columns_chosen] = self.priors_edges[columns_chosen]
        self._bpd2.update_channel_probs(updated_probs)
        # Luego le damos a decode.
        second_recovered_error = self._bpd2.decode(syndrome)
        if not self._bpd2.converge:
            # print('NO CONVERGENCE OG')
            return np.zeros(self._model.edge_observables_matrix.shape[0]), average_time
        second_recovered_error = (self._model.edge_observables_matrix @ second_recovered_error) % 2
        # if not np.all(error_edge == second_recovered_error):
        #     pass
        return second_recovered_error, average_time
    
    
    
if __name__ == "__main__":
    pass