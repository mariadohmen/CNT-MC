# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 19:31:00 2020

@author: Maria
"""

import random

import numpy as np

from scipy.constants import Boltzmann as kB


# Quantum Yields https://doi.org/10.1021/acs.jpclett.8b03732
# QY_E11_d = np.range(0.1,0.28)
# QY_E11 < 0.1

L_nm = 300  # length of the nanotube

# https://doi.org/10.1038/nphys1149
R_nm = 2  # radius of the exiton

N_DEF = 10  # number of defects per nanotube
T_STEP_ps = 1  # time step

# Bright exciton Diffusion Coefficient https://doi.org/10.1021/nn101612b
D_e_exc_nm_per_s = 1.07e15

# https://doi.org/10.1021/acsnano.6b02986
D_d_exc_nm_per_s = 07e15 / 3

# termal detrapping 10.1021/acs.jpclett.8b03732
k_dt_per_s = 0.5 * (1e12 / 385 + 1e12 / 1132) + 0.1e12 * np.exp(-1.6182e-11 /
                                                                (kB * 300))


#########
# Dependence of the lifetime on the defect density
#########

# https://doi.org/10.1021/acsnano.6b02986
TAU_e_ps = 10

TAU_d_ps = 82

TAU_b_ps = 85


def tau_func(n_defects, tau):
    return tau + 18.56986 * np.exp(-(n_defects + 0.05333) / 1.42987)


def k_nothing_e(t_step, n_defects, k_er, k_enr, k_ed, tau_e=TAU_e_ps):
    return (k_er + k_enr + k_ed) * tau_func(n_defects, tau_e) / t_step


def k_nothing_b(t_step, n_defects, k_br, k_bnr, k_be, tau_b=TAU_b_ps):
    return (k_bnr + k_br + k_be) * tau_func(n_defects, tau_b) / t_step


def k_nothing_d(t_step, n_defects, k_de, k_dnr, tau_d=TAU_d_ps):
    return (k_de + k_dnr) * tau_func(n_defects, tau_d) / t_step


def create_defects(CNT_length=L_nm, n_def=N_DEF):
    """Creates defects along the CNT at random position.

    Parameters
    ----------
    CNT_length : int, optional
        Length of the CNT in nm, global constant as default.
    n_def : int, optional
        Number of defects on the CNT, global constant as default.

    Returns
    -------
    pos_def : 1D array
        Positions in nm of the defects on the CNT stored in
        array size (n_def, 1)
    """
    return np.random.randint(0, CNT_length, size=n_def)


def create_exciton(CNT_length=L_nm):
    """Creates exciton on the CNT at random position.

    Parameters
    ----------
    CNT_length : int, optional
        Length of the CNT in nm, global constant as default.

    Returns
    -------
    pos_exc : int
        Position of the exciton along the CNT as a random integer."""
    return random.randrange(CNT_length)


def exciton_sim_4_lvl_full_exchange(t_step, kin_const, n_defects=N_DEF,
                                    CNT_length=L_nm, r_exc_nm=R_nm):
    """
    Simulation with three states above ground state: Excited state S11 (0),
    dark state (1) and bright state S11* (2). Diffusion along the nanotube is
    allowed for state 0 & 1. Exchange is possible between all states. The
    transition into the trap from 0 & 1 to 2 is modeled with MC steps, thermal
    detrapping is possible. Excitons are quenched if defects are too close
    together.

    Parameters
    ----------
    t_step : float
        Timestep in ps.
    constants : 1D array
        kinetic constants in order of:
        [k_br, k_er, k_bnr, k_enr, k_be, k_ed, k_de, k_dnr]
    n_defects : int, optional
        Number of defects on CNT. Default is 10.
    CNT_length : int, optional
        Length of the CNT in nm, global constant as default.
    r_exc_nm : int
        Radius of the Exciton in nm

    Returns
    -------
    exciton_fate : 1D array
        Array contains the binned fate of the exciton for each MC step:
            Array contains the binned fate of the exciton for each MC step:
            fate = 0 : E11* radiative decay (2)
            fate = 1 : E11 radiative decay (0)
            fate = 2 : E11* non-radiative decay (2)
            fate = 3 : E11 non-radiative decay (0)
            fate = 4 : Exciton escapes trap into exited state (0)
            fate = 5 : Exciton goes into dark state (1)
            fate = 6 : Exciton stays in bright state (2)
            fate = 7 : Free exciton diffusion (0 & 1)
            fate = 8 : Dark state non-radiative decay (1)
            fate = 9 : Exciton goes into excited state (0)
            fate = 10 : Exciton becomes trapped in bright state (2)
    """

    constants = np.zeros(11)
    constants[:6] = kin_const[:6]
    constants[-2] = kin_const[-1]
    constants[6] = k_nothing_b(t_step, n_defects, *kin_const[:6:2])
    constants[7] = k_nothing_e(t_step, n_defects, *kin_const[1:7:2])
    constants[-1] = kin_const[6]
    constants[8] = k_nothing_d(t_step, n_defects, *kin_const[-2:])

    # inital exciton is free, to 80 % in state dark state, to 20 % in exited
    # state
    fate = 7
    state = (np.random.random(1) < 0.8 ).astype(int)[0]

    # Initiate matrix to store exciton fate
    exciton_fate = np.zeros(len(constants))

    # Inital position of the exciton and defects
    pos_exc_0 = create_exciton(CNT_length)
    defects = create_defects(CNT_length, n_defects)

    # Masks defects which are too close together and result in non-radiative
    # decay
    defects = np.sort(defects)
    if len(defects) > 2:
        mask = [defects[1]-defects[0] >= r_exc_nm]
        mask.extend([True if defects[i+1]-defects[i] >= r_exc_nm
                     and defects[i]-defects[i-1] >= r_exc_nm
                     else False for i in np.arange(1, len(defects)-1)])
        mask.extend([defects[-1]-defects[-2] >= r_exc_nm])
    else:
        mask = np.ones(len(defects), dtype=bool)

    while fate > 3:

        # step if exciton is free
        if state == 0:
            pos_exc_1 = round(pos_exc_0 + (
                2 * D_e_exc_nm_per_s * t_step * 1e-12)**0.5)

        if state == 1:
            pos_exc_1 = round(pos_exc_0 + (
                2 * D_d_exc_nm_per_s * t_step * 1e-12)**0.5)

        # check if exciton became trapped
        if state < 2:
            pathway = np.arange(pos_exc_0, pos_exc_1)
            if np.in1d(pathway, defects).any():
                # set exciton to position of first encountered trap
                pos_exc_1 = defects[np.in1d(defects, pathway)][0]
                # check if non-radiative decay takes place due to close
                # proximity to another defect site
                if np.in1d(defects[mask], pos_exc_1).any():
                    fate = 10
                    state = 2
                    exciton_fate[fate] += 1
                else:
                    fate = 2
                    exciton_fate[fate] += 1
                    break

        # quenching of the exciton at tube end
        if pos_exc_1 >= CNT_length:
            fate = 3
            exciton_fate[fate] += 1
            break

        # fate of a trapped S11* exciton
        if state == 2:
            # calculate probability for fate of trapped exciton
            p_fate = np.array([e * random.uniform(0, 1)
                               for e in constants[:8:2]])
            # Store result for highest probability
            fate = 2*p_fate.argmax()
            exciton_fate[fate] += 1
            # if exciton escpaes move along, set state
            if fate == 4:
                state = 0
                pos_exc_1 += r_exc_nm

        # fate of S11 exciton
        elif state == 0:
            # calculate probability for fate of S11 exciton
            p_fate = np.array([e * random.uniform(0, 1)
                               for e in constants[1:9:2]])
            # Store result for highest probability
            fate = (p_fate.argmax() * 2 + 1)
            exciton_fate[fate] += 1
            if fate == 5:
                state = 1

        # fate of dark exciton
        else:
            p_fate = np.array([e * random.uniform(0, 1)
                               for e in constants[8:]])

            # Store result for highest probability
            fate = 7 + p_fate.argmax()
            exciton_fate[fate] += 1
            if fate == 8:
                exciton_fate[3] += 1
                break
            if fate == 9:
                state = 0

        # insurance that there won't be an endless loop
        if exciton_fate.sum() > 1e6:
            print('Simulation exceeds 1e6 steps, loop aborded')
            return exciton_fate

        # set position to new starting position
        pos_exc_0 = pos_exc_1

    return exciton_fate


def exciton_sim(t_step, kin_const, Diff_exc_e=D_e_exc_nm_per_s,
                Diff_exc_d=D_d_exc_nm_per_s, n_defects=N_DEF,
                CNT_length=L_nm, r_exc_nm=R_nm):
    """
    Simulation with three states above ground state: Excited state S11 (0),
    dark state (1) and bright state S11* (2). Diffusion along the nanotube is
    allowed for state 0 & 1. Exchange is possible between all states. The
    transition into the trap from 0 & 1 to 2 is modeled with MC steps, thermal
    detrapping is possible. Excitons are quenched if defects are too close
    together.

    Parameters
    ----------
    t_step : float
        Timestep in ps.
    constants : 1D array
        kinetic constants in order of:
        [k_br, k_er, k_bnr, k_enr, k_be, k_ed, k_de, k_dnr]
    n_defects : int, optional
        Number of defects on CNT. Default is 10.
    CNT_length : int, optional
        Length of the CNT in nm, global constant as default.
    r_exc_nm : int
        Radius of the Exciton in nm
    Diff_exc_e : float
        Diffusion constant for excited exciton, global constant as default.
    Diff_exc_d : float
        Diffusion constant for dark exciton, global constant as default.

    Returns
    -------
    exciton_fate : 1D array
        Array contains the binned fate of the exciton for each MC step:
            Array contains the binned fate of the exciton for each MC step:
            fate = 0 : E11* radiative decay (2)
            fate = 1 : E11 radiative decay (0)
            fate = 2 : E11* non-radiative decay (2)
            fate = 3 : E11 non-radiative decay (0)
            fate = 4 : Exciton escapes trap into exited state (0)
            fate = 5 : Exciton goes into dark state (1)
            fate = 6 : Exciton stays in bright state (2)
            fate = 7 : Free exciton diffusion (0 & 1)
            fate = 8 : Dark state non-radiative decay (1)
            fate = 9 : Exciton goes into excited state (0)
            fate = 10 : Exciton becomes trapped in bright state (2)
    """

    constants = np.zeros(11)
    constants[:6] = kin_const[:6]
    constants[-2] = kin_const[-1]
    constants[6] = k_nothing_b(t_step, n_defects, *kin_const[:6:2])
    constants[7] = k_nothing_e(t_step, n_defects, *kin_const[1:7:2])
    constants[-1] = kin_const[6]
    constants[8] = k_nothing_d(t_step, n_defects, *kin_const[-2:])

    # inital exciton is free, to 80 % in state dark state, to 20 % in exited
    # state
    fate = 7
    state = (np.random.random(1) < 0.8).astype(int)[0]

    # Initiate matrix to store exciton fate
    exciton_fate = np.zeros(len(constants))

    # Inital position of the exciton and defects
    pos_exc_0 = create_exciton(CNT_length)
    defects = create_defects(CNT_length, n_defects)

    # Masks defects which are too close together and result in non-radiative
    # decay
    defects = np.sort(defects)
    if len(defects) > 2:
        mask = [defects[1]-defects[0] >= r_exc_nm]
        mask.extend([True if defects[i+1]-defects[i] >= r_exc_nm
                     and defects[i]-defects[i-1] >= r_exc_nm
                     else False for i in np.arange(1, len(defects)-1)])
        mask.extend([defects[-1]-defects[-2] >= r_exc_nm])
    else:
        mask = np.ones(len(defects), dtype=bool)

    while fate > 3:

        # step if exciton is free
        if state == 0:
            pos_exc_1 = round(pos_exc_0 + (
                2 * Diff_exc_e * t_step * 1e-12)**0.5)

        if state == 1:
            pos_exc_1 = round(pos_exc_0 + (
                2 * Diff_exc_d * t_step * 1e-12)**0.5)

        # check if exciton became trapped
        if state < 2:
            pathway = np.arange(pos_exc_0, pos_exc_1)
            if np.in1d(pathway, defects).any():
                # set exciton to position of first encountered trap
                pos_exc_1 = defects[np.in1d(defects, pathway)][0]
                # check if non-radiative decay takes place due to close
                # proximity to another defect site
                if np.in1d(defects[mask], pos_exc_1).any():
                    fate = 10
                    state = 2
                    exciton_fate[fate] += 1
                else:
                    fate = 2
                    exciton_fate[fate] += 1
                    break

        # quenching of the exciton at tube end
        if pos_exc_1 >= CNT_length:
            fate = 3
            exciton_fate[fate] += 1
            break

        # fate of a trapped S11* exciton
        if state == 2:
            # calculate probability for fate of trapped exciton
            p_fate = np.array([e * random.uniform(0, 1)
                               for e in constants[:8:2]])
            # Store result for highest probability
            fate = 2*p_fate.argmax()
            exciton_fate[fate] += 1
            # if exciton escpaes move along, set state
            if fate == 4:
                state = 0
                pos_exc_1 += r_exc_nm

        # fate of S11 exciton
        elif state == 0:
            # calculate probability for fate of S11 exciton
            p_fate = np.array([e * random.uniform(0, 1)
                               for e in constants[1:9:2]])
            # Store result for highest probability
            fate = (p_fate.argmax() * 2 + 1)
            exciton_fate[fate] += 1
            if fate == 5:
                state = 1

        # fate of dark exciton
        else:
            p_fate = np.array([e * random.uniform(0, 1)
                               for e in constants[8:]])

            # Store result for highest probability
            fate = 7 + p_fate.argmax()
            exciton_fate[fate] += 1
            if fate == 8:
                exciton_fate[3] += 1
                break
            if fate == 9:
                state = 0

        # insurance that there won't be an endless loop
        if exciton_fate.sum() > 1e6:
            print('Simulation exceeds 1e6 steps, loop aborded')
            return exciton_fate

        # set position to new starting position
        pos_exc_0 = pos_exc_1

    return exciton_fate
