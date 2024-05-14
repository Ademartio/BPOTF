import numpy as np
from ldpc import bp_decoder
import copy
import time



class UF:
    
    
    def __init__(self,
                 H: np.array,
                 p : float,
                 ):
        self.H = H
        self.p = p
        self._bpd = bp_decoder(
            H,
            error_rate = p
        )
        self._bpd2 = bp_decoder(
            H,
            error_rate = p
        )
        self.rows = H.shape[0]
        self.columns = H.shape[1]
        
        self.Hog = copy.deepcopy(self.H)
        
        zeros_rows = np.zeros((2, self.columns))
        self.Hog =  np.vstack((self.Hog, zeros_rows))
        self.Hog = self.Hog.astype(int)
        
        # We add virtual checks so all columns have at least two non-trivial elements.
        for i in range(self.columns):
            ones_in_col = np.where(H[:,i] == 1)[0]
            if len(ones_in_col) == 1:
                if ones_in_col[0] < self.rows//2:
                    self.Hog[-2,i] = 1
                else:
                    self.Hog[-1,i] = 1
    
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
        sorted_indices = np.argsort(llrs)
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
        cluster_array = self.SetCluster()
        columns_chosen = np.zeros(self.columns, dtype = int)
        for column in range(self.columns):
            # Checker nos sirve para ever que checks coge cada evento.
            
            checker = np.zeros(self.rows+2, dtype = int)
            # Depths: (root, depth, boolean about increasing tree size)
            depths = [0, -1, False]
            boolean_condition = True
            non_trivial_elements_in_column = np.where(self.Hog[:,sorted_indices[column]]==1)[0]
            for non_trivial_element in non_trivial_elements_in_column:
                root, depth = self.Find(non_trivial_element, cluster_array)
                if checker[root] == 1:
                    boolean_condition = False
                    break
                checker[root] = 1
                if depth > depths[1]:
                    depths = [root, depth, False]
                if depth == depths[1]:
                    depths[2] = True
            if boolean_condition:
                non_trivial_checker = np.where(checker == 1)[0]
                for element in non_trivial_checker:
                    cluster_array[element,0] = depths[0]
                columns_chosen[sorted_indices[column]] = 1
                if  depths[2] == 1:
                    cluster_array[depths[0]][1] += 1
        indices_columns_chosen = np.where(columns_chosen==1)[0]
        # returned_H = H_sorted[:-2,indices_columns_chosen]
        # new_rows, new_cols = returned_H.shape
        # if new_rows >= new_cols:
        #     returned_H = np.hstack((returned_H,np.zeros((new_cols, new_rows-new_cols+1))))
        return indices_columns_chosen
                
    def decode(self, syndrome : np.array):
        
        recovered_error = self._bpd.decode(syndrome)
        
        if self._bpd.converge:
            return recovered_error, 0
        
        llrs = self._bpd.log_prob_ratios
        sorted_indices = self.sort_matrix(llrs)
        a = time.perf_counter()
        columns_chosen = self.Kruskal_hypergraph(sorted_indices)
        b = time.perf_counter()
        average_time = b-a
        # _bpd_squared = bp_decoder(
        #     H_squared,
        #     error_rate=self.p,
        # )
        updated_probs = np.zeros(self.columns)
        updated_probs[columns_chosen] = self.p
        self._bpd2.update_channel_probs(updated_probs)
        second_recovered_error = self._bpd2.decode(syndrome)
        if not self._bpd2.converge:
            print('Main error')
        # self._bpd.update_channel_probs(np.full(self.columns, self.p))
        # non_trivials = sorted_indices[columns_chosen[np.where(second_recovered_error == 1)[0]].astype(int)]
        # second_recovered_error_n = np.zeros(self.columns, dtype = bool)
        # second_recovered_error_n[non_trivials] = True
        
        return second_recovered_error, average_time
    
    
    
if __name__ == "__main__":
    pass