from qecsim.models.rotatedplanar import RotatedPlanarCode
from ldpc import bp_decoder
import numpy as np
import random
from qecsim import paulitools as pt
from ldpc import bposd_decoder

def error_generation(p, n):
    """Depolarizing error generation

    Args:
        p (float): Probability of error
        n (int): number of qubits
    """
    error = np.zeros(2*n)
    probabilities = [(1 - p), p/3, p/3, p/3]
    results = ['I', 'X', 'Y', 'Z']

    # Generate n random realizations based on the probabilities
    realizations = random.choices(results, probabilities, k=n)
    for index, realization in enumerate(realizations):
        if realization == 'X':
            error[index] = 1
        elif realization == 'Z':
            error[index+n] = 1
        elif realization == 'Y':
            error[index] = 1
            error[index+n] = 1
    return error.astype(int)

def order_matrix_by_vector(vector, matrix):
    # Sort the indices of the vector based on its values
    sorted_indices = np.argsort(vector)
    # Reorder the columns of the matrix based on the sorted indices
    ordered_matrix = matrix[:, sorted_indices]
    return ordered_matrix, sorted_indices

def kruskal_on_hypergraph(Hog):
    """
    We iterate over all the columns of the matrix of ordered llr columns and keep the linearly independent ones.
    We do so by choosing ones which do not produce loops.

    Args:
        Hog (np.array): Ordered pcm.

    Returns:
        matrix_with_empty_column (np.array): Matrix of linear independent columns plus an empty column.
        column_to_square (np.array): Vector containing the positions of the columns that have been chosen as l.i.
    """
     
    rows, columns = Hog.shape
    column_to_square = np.zeros(rows, dtype = int)
    
    zeros_rows = np.zeros((2, columns))
    H =  np.vstack((Hog, zeros_rows))
    
    # We add virtual checks so all columns have at least two non-trivial elements.
    for i in range(columns):
        ones_in_col = np.where(H[:,i] == 1)[0]
        if len(ones_in_col) == 1:
            if ones_in_col[0] < rows//2:
                H[-2,i] = 1
            else:
                H[-1,i] = 1
           
    # cluster_array indicates in which tree each node is. Tree 0 means they are in no  t ree.     
    cluster_array = np.zeros(H.shape[0], dtype = int)
    cluster_number = 1
    column_number = 1
    
    # First column is always l.i.
    cluster_array[np.where(H[:,0] == 1)[0]] = cluster_number
    cluster_number += 1
    
    for i in range(1,columns):
        
        cluster_values = cluster_array[np.where(H[:,i] == 1)[0]]
        non_zero_cluster_values = cluster_values[np.nonzero(cluster_values)]
        if np.any(np.bincount(non_zero_cluster_values) > 1):
            # Loop, omit column.
            continue

        if len(non_zero_cluster_values) == 0:
            # New tree
            cluster_array[(np.where(H[:,i] == 1)[0])] = cluster_number
            cluster_number += 1
        
        elif len(non_zero_cluster_values) == 1:
            # Absortion of the node into a former tree
            cluster_array[(np.where(H[:,i] == 1)[0])] = non_zero_cluster_values[0]

        else:
            # UNION of several trees through the new node
            cluster_array[(np.where(H[:,i] == 1)[0])] = cluster_number
            cluster_array[np.where(np.isin(cluster_array, non_zero_cluster_values))] = cluster_number
            cluster_number += 1

        column_to_square[column_number] = i
        column_number += 1
        if column_number == rows:
            break
    

    assert column_to_square[-1] != 0, " Ha habido un error, no ha encontrado n-k columnas independientes"
    
    H_square = H[:,column_to_square.astype(int)]
    
    empty_column = np.zeros((H_square.shape[0], 1))  # Create an empty column filled with zeros
    matrix_with_empty_column = np.hstack((H_square, empty_column))

    return matrix_with_empty_column[:-2,:], column_to_square
    
        
    
if __name__ == "__main__":
    distances = [3,5, 7, 9]
    NMCs = [10**4, 10**4, 10**4, 10**4, 10**4, 10**4, 10**4, 10**4, 10**4, 10**3, 10**3, 10**3, 10**3]
    ps = np.linspace(0.01, 0.13, num=13)
    PlsBP = {}
    PlsBPOSD = {}
    PlsBPBP = {}

    for distance in distances:
        myCode = RotatedPlanarCode(distance, distance)
        pcm = np.zeros((myCode.stabilizers.shape[0], myCode.stabilizers.shape[1]), dtype = int)

        pcm[:,:pcm.shape[1]//2] = myCode.stabilizers[:,pcm.shape[1]//2:]
        pcm[:,pcm.shape[1]//2:] = myCode.stabilizers[:,:pcm.shape[1]//2]
        
        PlsBP[distance] = []
        PlsBPBP[distance] = []
        PlsBPOSD[distance] = []
        print('\n')
        print(f'Distance: {distance}')
        print('-------------------------------------------------')
        for index, p in enumerate(ps):
        
            _bp = bp_decoder(
                pcm,
                max_iter=30,
                error_rate = p
            )
            
            _bposd = bposd_decoder(
                pcm,
                max_iter=30,
                error_rate = p,
                osd_method = "osd_0"
            )
            
            PlBP = 0
            PlBPBP = 0
            PlBPOSD = 0
            
            for iteration in range(NMCs[index]):
                PlBOSDfailed = False
                
                error = error_generation(p, myCode.n_k_d[0])
                syndrome = pt.bsp(error, myCode.stabilizers.T)
                # syndrome = pt.bsp(error, pcm.T)
                # BPOSD decoder
                #----------------------------
                recovered_error_BPOSD = _bposd.decode(syndrome)
                if np.any(pt.bsp(recovered_error_BPOSD ^ error, myCode.logicals.T) == 1):
                    PlBPOSD += 1/NMCs[index]
                    PlBOSDfailed = True
                #----------------------------
                
                recovered_error = _bp.decode(syndrome)
                if _bp.converge:
                    if np.any(pt.bsp(recovered_error ^ error, myCode.logicals.T) == 1):
                        PlBP += 1/NMCs[index]
                        PlBPBP += 1/NMCs[index]
                    continue
                else:
                    PlBP += 1/NMCs[index]

                llrs = _bp.log_prob_ratios

                ordered_pcm, sorted_indices = order_matrix_by_vector(llrs, pcm)
                pcm_squared, columns_chosen = kruskal_on_hypergraph(ordered_pcm)
                
                _bp_squared = bp_decoder(
                    pcm_squared,
                    max_iter=30,
                    error_rate = p
                )
                
                second_recovered_error = _bp_squared.decode(syndrome)
            
                if _bp_squared.converge:
                    non_trivials = sorted_indices[columns_chosen[np.where(second_recovered_error == 1)[0]].astype(int)]
                    second_recovered_error_n = np.zeros(2*myCode.n_k_d[0])
                    second_recovered_error_n[non_trivials] = 1
                    if np.any(pt.bsp(second_recovered_error_n.astype(int) ^ error, myCode.logicals.T) == 1):
                        PlBPBP += 1/NMCs[index]
                        if not PlBOSDfailed:
                            print(2)
                            print('Error')
                            print(np.where(error == 1)[0])
                            print('BPOSD response')
                            print(np.where(recovered_error_BPOSD == 1)[0])
                            print('BPBP response')
                            print(np.where(second_recovered_error_n == 1)[0])
                        
                else:
                    PlBPBP += 1/NMCs[index]
                    print(error)
                    print(syndrome)
                    print(columns_chosen)
                    print(1)
                    continue
                
            
            PlsBP[distance].append(PlBP)
            PlsBPOSD[distance].append(PlBPOSD)
            PlsBPBP[distance].append(PlBPBP)
                
            print(f'Physical error: {p}')
            print(f'Error BP: {PlBP}')
            print(f'Error BPOSD: {PlBPOSD}')
            print(f'Error BPBP: {PlBPBP}')
            print('\n')
