import scipy
import numpy as np
from algorithm2 import error_generation
from ldpc import bp_decoder, bposd_decoder
from UF import UF
from error import classical_error
import time


if __name__ == "__main__":
    hx = scipy.sparse.load_npz("BPBP/data_noise/pcms/hx_400_16_6.npz").astype(int).toarray()
    lx = scipy.sparse.load_npz("BPBP/data_noise/pcms/lx_400_16_6.npz").astype(int).toarray()
    
    distance = 3
    NMCs = [10**4, 10**4, 10**4, 10**4, 10**4, 10**4, 10**4, 10**4, 10**4, 10**4, 10**4, 10**4, 10**4]
    ps = np.linspace(0.005, 0.015, num=13)
    PlsBP = {}
    PlsBPOSD = {}
    PlsBPBP = {}
    columns = hx.shape[1]

    pcm = hx

    PlsBP[distance] = []
    PlsBPBP[distance] = []
    PlsBPOSD[distance] = []
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
        
        _uf = UF(
            pcm,
            p
        )
        
        PlBP = 0
        PlBPBP = 0
        PlBPOSD = 0
        time_av_BPOSD = 0
        time_av_BPBP = 0
        kruskal_time_list = []
        
        for iteration in range(NMCs[index]):
            bposdbool = True
            bpbpbool = True
            error = classical_error(p, columns)
            # error = np.zeros(columns)
            # error_positions = np.array([51,115,211])
            # error[error_positions] = 1
            syndrome = np.dot(hx, error) % 2
            # syndrome = pt.bsp(error, hx.T)
            # syndrome = pt.bsp(error, pcm.T)
            # BPOSD decoder
            #----------------------------
            a = time.time()
            recovered_error_BPOSD = _bposd.decode(syndrome)
            b = time.time()
            time_av_BPOSD += (b-a)/NMCs[index]
            recovered_error = _bp.decode(syndrome)
            a = time.time()
            recovered_error_BPBP, kruskal_time = _uf.decode(syndrome)
            b = time.time()
            time_av_BPBP += (b-a)/NMCs[index]
            
            if np.any(np.dot(recovered_error_BPOSD ^ error.astype(int), lx.T) %2 == 1):
                PlBPOSD += 1/NMCs[index]
                bposdbool = False
                
            if np.any(np.dot(recovered_error_BPBP ^ error.astype(int), lx.T) % 2 == 1):
                PlBPBP += 1/NMCs[index]
                bpbpbool = False
            #----------------------------
            if bposdbool != bpbpbool:
                pass
            if _bp.converge:
                if np.any(np.dot(recovered_error ^ error.astype(int), lx.T)%2 == 1):
                    PlBP += 1/NMCs[index]
                continue
            else:
                PlBP += 1/NMCs[index]

            if kruskal_time > 0:
                kruskal_time_list.append(kruskal_time)
            
            # print(iteration)
            
        
        PlsBP[distance].append(PlBP)
        PlsBPOSD[distance].append(PlBPOSD)
        PlsBPBP[distance].append(PlBPBP)
            
        print(f'Physical error: {p}')
        print(f'Error BP: {PlBP}')
        print(f'Error BPOSD: {PlBPOSD} with average time {time_av_BPOSD}')
        print(f'Error BPBP: {PlBPBP} with average time {time_av_BPBP}')
        if len(kruskal_time_list) > 0:
            print(f'Average Kruskal time {sum(kruskal_time_list)/len(kruskal_time_list)}')
        print('\n')