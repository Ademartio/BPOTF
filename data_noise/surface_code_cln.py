from qecsim.models.rotatedplanar import RotatedPlanarCode
from ldpc import bp_decoder
import numpy as np
import random
from qecsim import paulitools as pt
# from networkx import find_cycle
import networkx as nt
from copy import deepcopy
from ldpc import bposd_decoder
from UFCLN import UFCLN
from UF import UF
import time
from pcms.BBcode import bbpcm
from scipy.linalg import null_space
from error import classical_error
import stim
from beliefmatching import detector_error_model_to_check_matrices
import csv
from tqdm import tqdm



    
def cln_surface_code(distance):

    """We will use  stim for computing the pcm from the surface code under cln. Then, we attempt to compute
    it with BPOSD and BPBP.
    """
    
    p = 0.001 # .5%
    run_count = 10**4
    
    # seed = np.random.randint(2e9)
    max_iter = 30
    
    surface_code_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=distance,
        distance=distance,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p
    )

    dem = surface_code_circuit.detector_error_model(decompose_errors=True)
    DEM = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=False)
    
    pcm = DEM.check_matrix.toarray()
    obs_matrix = DEM.observables_matrix.toarray()
    edge_obs_matrix = DEM.edge_observables_matrix.toarray()
    priors = DEM.priors
    
    decoderBPOSD = bposd_decoder(pcm, channel_probs=priors, max_iter=max_iter, bp_method="ps", ms_scaling_factor = 0.625, osd_method = "osd0")
    decoderBPBP = UFCLN(
                dem
            )
    decoderBPBPog = UF(
        pcm,
        priors
    )
    
    sampler = surface_code_circuit.compile_detector_sampler()
    
    detection_events, observable_flips = sampler.sample(run_count, separate_observables=True)
    
    PlBPOSD = 0
    PlBPBP = 0
    PlBPBP2 = 0
    time_1 = 0
    time_2 = 0
    
    for index, detection_event in enumerate(tqdm(detection_events)):
        a = time.time()
        recovered_1 = decoderBPOSD.decode(detection_event)
        b = time.time()
        time_1 += b-a
        if not np.all(observable_flips[index] == ((obs_matrix @ recovered_1) % 2)):
            PlBPOSD += 1
        a = time.time()
        recovered_2 = decoderBPBP.decode(detection_event)
        b = time.time()
        time_2 += b-a
        if not np.all(observable_flips[index] == recovered_2[0]):
            PlBPBP += 1
        recovered_3 = decoderBPBPog.decode(detection_event)
        if not np.all(observable_flips[index] == ((obs_matrix @ recovered_3[0]) % 2)):
            PlBPBP2 += 1
            
    print(f"\n Surface code cln {distance}\n.................................")
    print(f'Pl BPOSD = {PlBPOSD/run_count} with average time {time_1/run_count}')
    print(f'Pl BPBP = {PlBPBP/run_count} with average time {time_2/run_count}')
    print(f'Pl BPBP = {PlBPBP2/run_count}')
    
    
    
if __name__ == "__main__":
    for distance in [3,5,7,9]:
        cln_surface_code(distance)