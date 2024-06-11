from qecsim.models.rotatedplanar import RotatedPlanarCode
from ldpc import bp_decoder
import numpy as np
import random
from quasicyclic import quasi_cyclic_from_params, QuasiCyclic
from quasicyclic.quasi_cyclic_phenom import quasi_cyclic_from_params_phenom, QuasiCyclic_phenom
from decoders.preprocessing import preprocessing
import networkx as nt
from copy import deepcopy
from ldpc import bposd_decoder
import time
from scipy.linalg import null_space
import stim
from beliefmatching import detector_error_model_to_check_matrices
import csv
from tqdm import tqdm
import pymatching
from decoders.UFCLN import UFCLN



def cln_BB_code(p, l, m=6):

    """We will use  stim for computing the pcm from the surface code under cln. Then, we attempt to compute
    it with BPOSD and BPBP.
    """
    
    p = 0.001 # .5%
    run_count = 10**4
    
    # seed = np.random.randint(2e9)
    max_iter = 30
    
    
    qc: QuasiCyclic = quasi_cyclic_from_params(l=l, m=6, A_poly="x^3 + y + y^2", B_poly="y^3 + x + x^2")
    circuit: stim.Circuit = qc.generate_circuit(measure_basis="Z", num_rounds=10, p=p)
    
    qc_2: QuasiCyclic_phenom = quasi_cyclic_from_params_phenom(l=l, m=6, A_poly="x^3 + y + y^2", B_poly="y^3 + x + x^2")
    circuit_2: stim.Circuit = qc_2.generate_circuit(measure_basis="Z", num_rounds=10, p=p)
    
    demcln = circuit.detector_error_model(decompose_errors=False)
    demphen = circuit_2.detector_error_model(decompose_errors=False)
    
    # We map the demcln matrix to the demphen p matrix.
    A = preprocessing(demcln, demphen)

    dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures= True)
    DEM = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=False)
    
    pcm = DEM.check_matrix.toarray()
    rank = np.linalg.matrix_rank(pcm)
    print(f'Rank of the pcm matrix {rank}')
    # obs_matrix = DEM.observables_matrix.toarray()
    # edge_obs_matrix = DEM.edge_observables_matrix.toarray()
    priors = DEM.priors
    
    decoderBPOSD = bposd_decoder(
        pcm,
        channel_probs=priors,
        max_iter=max_iter, bp_method="ps",
        ms_scaling_factor = 0.625,
        osd_method = "osd0"
    )
    decoderBPBP = UFCLN(
        dem,
        distance
    )
    
    
    
    
    sampler = circuit.compile_detector_sampler()
    
    detection_events, observable_flips = sampler.sample(run_count, separate_observables=True)
    
    PlBPOSD = 0
    PlBPBP = 0
    PlBPBP2 = 0
    PlMWPM = 0
    time_1 = 0
    time_2 = 0
    
    for index, detection_event in enumerate(detection_events):
        # error = generate_binary_array(DEM.priors)
        # logical_error = (DEM.observables_matrix @ error) % 2
        # syndrome = (pcm @ error) % 2
        a = time.time()
        recovered_1 = decoderBPOSD.decode(detection_event)
        b = time.time()
        time_1 += b-a
        if not np.all(observable_flips[index] == (DEM.observables_matrix @ recovered_1) % 2):
            PlBPOSD += 1
        a = time.time()
        recovered_2 = decoderBPBP.decode(detection_event)
        b = time.time()
        time_2 += b-a
        if not np.all(observable_flips[index] == recovered_2[0]):
            PlBPBP += 1

            
    print(f"\n Surface code cln {distance}\n.................................")
    print(f'Pl BPOSD = {PlBPOSD/run_count} with average time {time_1/run_count}')
    print(f'Pl BPBP = {PlBPBP/run_count} with average time {time_2/run_count}')
    print(f'Pl BPBP naive = {PlBPBP2/run_count}')
    print(f'Pl MWPM = {PlMWPM/run_count}')
    
    
if __name__ == "__main__":
    
    
    
    for distance in [6,9,12]:
        cln_BB_code(l=distance)