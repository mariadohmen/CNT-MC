# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:52:22 2020

@author: student
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

#scipy constants
from scipy.constants import Boltzmann as kB
from scipy.constants import Avogadro as NA
from scipy.constants import Planck as h
from scipy.constants import speed_of_light as c0
from scipy.constants import R

sys.path.append('../CNT-SIM/')
from CNTSim_server import CNTSimFile
from mc_sim import exciton_sim

k_er_per_s = 1e8
k_br_per_s = 1e8  # constant for radiativ decay from S11*
k_enr_per_s = 2e8
k_bnr_per_s = 4e8   # constant for non-radiativ decay from S11*

k_de_per_s = 1e8  # constant for going from dark to S11 state
k_ed_per_s = 1e8  # constant for going from S11 to dark statestate

# termal detrapping 10.1021/acs.jpclett.8b03732
k_bd_per_s = 0.5 * (1e12 / 385 + 1e12 / 1132) + 0.1e12 * np.exp(-1.6182e-11 /
                                                                (kB * 300))
k_dnr_per_s = 2e8

kin_const_1 = np.array([k_br_per_s, k_er_per_s, k_bnr_per_s, k_enr_per_s,
                        k_bd_per_s, k_ed_per_s, k_de_per_s, k_dnr_per_s])

diff_const = np.zeros((10, 2))
diff_const[:, 0] = np.linspace(1.07e15, 1.07e15 * 2, 10)
diff_const[:, 1] = np.linspace(1.07e15/3, 1.07e15/3 * 2, 10)

constants_array = np.zeros((10, 2))
constants_array[:, 0] = np.linspace(4e8, 4e8 * 3, 10)
constants_array[:, 1] = np.linspace(2e8, 2e8 / 2, 10)


for i in np.arange(10):
    sim = CNTSimFile(
            f'../sim_output/2020-10-21-exciton_sim-kenr_kbnr_diff_sever_8_defect_{i}.h5',
            kin_const_1)
    sim.parameter_dependence(100000, exciton_sim, constants_array, ['k_bnr',
                                                                    'k_enr'],
                             ['k_br', 'k_er', 'k_bnr', 'k_enr', 'k_be', 'k_ed',
                              'k_de', 'k_dnr'],
                             diff_const,
                             (1.07e15, 1.07e15/3),
                             {'t_step': 1, 'r_exc_nm': 2, 'n_defects': 8,
                              'CNT_length': 750})
    sim.save()
