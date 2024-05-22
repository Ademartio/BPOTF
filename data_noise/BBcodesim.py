from qecsim.models.rotatedplanar import RotatedPlanarCode
from ldpc import bp_decoder
import numpy as np
import random
from qecsim import paulitools as pt
# from networkx import find_cycle
import networkx as nt
from copy import deepcopy
from ldpc import bposd_decoder
from UF import UF
import time
from pcms.BBcode import bbpcm
from scipy.linalg import null_space
from error import classical_error

import csv



    
if __name__ == "__main__":

    distances = [12]
    NMCs = [10**4, 10**4, 10**4, 10**4, 10**4, 10**4, 10**4, 10**4, 10**4, 10**4, 10**4, 10**4, 10**4]
    ps = np.linspace(0.01, 0.08, num=13)
    PlsBP = {}
    PlsBPOSD = {}
    PlsBPBP = {}

    times_BPOSD = {}
    times_BPBP = {}

    f = open("times.csv", "w")

    for distance in distances:
        
        
        pcm, Hx, Hz = bbpcm(distance,6,[[0,3],[1,1],[1,2]],[[1,3],[0,1],[0,2]])
        
        rows, columns = Hx.shape
        
        logicals_z = null_space(Hx)
        logicals_x = null_space(Hz)
        
        PlsBP[distance] = []
        PlsBPBP[distance] = []
        PlsBPOSD[distance] = []

        times_BPOSD[distance] = []
        times_BPBP[distance] = []

        print(f'Distance: {distance}')
        print('-------------------------------------------------')
        for index, p in enumerate(ps):
        
            _bp = bp_decoder(
                Hx,
                max_iter=30,
                error_rate = p
            )
            
            _bposd = bposd_decoder(
                Hx,
                max_iter=30,
                error_rate = p,
                osd_method = "osd_0"
            )
            
            _uf = UF(
                Hx,
                p
            )
            
            PlBP = 0
            PlBPBP = 0
            PlBPOSD = 0
            time_av_BPOSD = 0
            time_av_BPBP = 0
            kruskal_time_list = []
            
            for iteration in range(NMCs[index]):
                error = classical_error(p, columns)
                # error = np.zeros(myCode.n_k_d[0]*2, dtype = int)
                # error[3] = 1
                syndrome = np.dot(Hx,error) %2
                # syndrome = pt.bsp(error, pcm.T)
                # BPOSD decoder
                #----------------------------
                a = time.time()
                recovered_error_BPOSD = _bposd.decode(syndrome)
                b = time.time()
                time_av_BPOSD += (b-a)/NMCs[index]
                times_BPOSD[distance].append(b-a)
                recovered_error = _bp.decode(syndrome)
                a = time.time()
                recovered_error_BPBP, kruskal_time = _uf.decode(syndrome)
                b = time.time()
                time_av_BPBP += (b-a)/NMCs[index]
                times_BPBP[distance].append(b-a)
                
                
                if (not np.all((np.dot(logicals_x.T, (recovered_error_BPOSD ^ error)) % 2).astype(int) == 0)):
                    PlBPOSD += 1/NMCs[index]
                    
                    
                if (not np.all((np.dot(logicals_x.T, (recovered_error_BPBP ^ error)) % 2).astype(int) == 0)):
                    PlBPBP += 1/NMCs[index]
                    
                #----------------------------
                
                if _bp.converge:
                    if (not np.all((np.dot(logicals_x.T, (recovered_error ^ error)) % 2).astype(int) == 0)):
                        PlBP += 1/NMCs[index]
                    continue
                else:
                    PlBP += 1/NMCs[index]

                if kruskal_time > 0:
                    kruskal_time_list.append(kruskal_time)
                
            
            PlsBP[distance].append(PlBP)
            PlsBPOSD[distance].append(PlBPOSD)
            PlsBPBP[distance].append(PlBPBP)
                
            print(f'Physical error: {p}')
            print(f'Error BP: {PlBP}')
            print(f'Error BPOSD: {PlBPOSD} with average time {time_av_BPOSD}')
            print(f'Error BPBP: {PlBPBP} with average time {time_av_BPBP}')
            
            f.write(f'Distance: {distance} and physical error: {p}\n')
            f.write('-------------------------------------------------\n')
            f.write('Max BPOSD time: {}\n'.format(max(times_BPOSD[distance])))
            f.write('Max BPBP time: {}\n\n'.format(max(times_BPBP[distance])))
            
            if len(kruskal_time_list) > 0:
                print(f'Average Kruskal time {sum(kruskal_time_list)/len(kruskal_time_list)}')
            print('\n')
        
    f.close()