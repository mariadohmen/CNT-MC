# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:48:26 2020

@author: Maria
"""

import random

import numpy as np

from ipywidgets.widgets import Text

from scipy.constants import Boltzmann as kB

L_nm = 300  # length of the nanotube
R_nm = 1  # radius of the exiton
N_DEF = 10  # number of defects per nanotube
T_STEP_ps = 1  # time step

# Exciton Diffusion Coefficient https://doi.org/10.1021/nn101612b
D_exc_nm_per_s = 1.07e15

k_r_per_s = 1e10  # radiativ decay constant
k_nr_per_s = 1e11  # non-radiativ decay constant bright
k_d_per_s = 1e11  # constant for transi on into dark state
k_n_per_s = 1e11  # exciton is stable


def exciton_fate():
    """Determines the fate of a single exciton.

    Returns
    -------
    exciton_fate : 1D array
        Matrix showing the binned fate of the MC steps. Order is
        k_r_per_s, k_nr_per_s, k_d_per_s, k_d_per_s
    """
    kin_const = [k_r_per_s, k_nr_per_s, k_d_per_s, k_d_per_s]
    fate = 3  # fate of single exciton in MC step
    exciton_fate = np.zeros(len(kin_const))

    while fate > 1:
        # calculate the probability for exciton decay
        p_fate = np.array([e * random.uniform(0, 1) for e in kin_const])

        # Store result for highest probability
        fate = p_fate.argmax()
        exciton_fate[fate] += 1

        # insurance that there won't be an endless loop
        if exciton_fate.sum() > 1e3:
            print('Simulation exceeds 1e3 steps, loop aborded')
            return exciton_fate
    return exciton_fate


def photons_fate(n_photons, func, func_kwargs={}):
    """Simulate the fate of numerous photons.
    For a key to the fate look a the respective exciton fate function.

    Parameters
    ----------
    n_photons : int
        Number of photons to be used in the simulation.
    func : callable
        Function to perform MC simulation of a single exciton.
    func_kwargs: dict
        Dictionary containing the arguments and keyword arguments for
        func.

    Returns
    -------
    photons_fate : 1D array
        Binnend fate of the photons in simulation.
    quantum_yield : 1D array
        Quantum yield of the non-radiative and radiative decay."""
    info = Text()
    display(info)

    # initiate matrix size
    photons_fate = func(**func_kwargs)
    info.value = f"Processing photon (1/ {n_photons})"

    # loop for the desired number of photons
    for p in np.arange(n_photons-1):
        photons_fate += func(**func_kwargs)
        info.value = f"Processing photon ({p+2}/ {n_photons})"

    # calculate the quantum yield
    quantum_yield = photons_fate[:2] / n_photons

    return photons_fate, quantum_yield


# termal detrapping 10.1021/acs.jpclett.8b03732
k_dt_per_s = 0.5 * (1e12 / 385 + 1e12 / 1132) + 0.1e12 * np.exp(-1.6182e-11 /
                                                                (kB * 300))

TAU_ps = 100
TAU_d_ps = 1000

k_r_per_s = 1.5e10  # constant for radiativ decay from E11
k_d_r_per_s = 2.5e09  # constant for radiativ decay from E11*
k_nr_per_s = 5e09  # constant of non-radiativ decay from E11
k_d_nr_per_s = 5e09  # constant for non-radiativ decay from E11*

KIN_CONST = np.array([k_d_r_per_s, k_r_per_s, k_d_nr_per_s, k_nr_per_s, k_dt_per_s])


def k_nothing(t_step, k_r=k_r_per_s, k_nr=k_nr_per_s, tau=TAU_ps):
    return (k_r + k_nr) * tau / t_step


def k_nothing_d(t_step, k_d_r=k_d_r_per_s, k_d_nr=k_d_nr_per_s,
                k_dt=k_dt_per_s, tau_d=TAU_d_ps):
    return (k_d_r + k_d_nr + k_dt) * tau_d / t_step


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


def exciton_walk(t_step, kin_const, n_defects=10, CNT_length=L_nm,):
    """
    Parameters
    ----------
    t_step : float
        Timestep in ps.
    n_defects : int, optional
        Number of defects on CNT. Default is 10.
    CNT_length : int, optional
        Length of the CNT in nm, global constant as default.
    constants : 1D array
        kinetic constants in order of:
        [k_d_r_per_s, k_r_per_s, k_d_nr_per_s, k_nr_per_s, k_dt_per_s]

    Returns
    -------
    exciton_fate : 1D array
        Array contains the binned fate of the exciton for each MC step:
        fate = 0 : E11* radiative decay
        fate = 1 : E11 radiative decay
        fate = 2 : E11* non-radiative decay
        fate = 3 : E11 non-radiative decay
        fate = 4 : Exciton stays in trap
        fate = 5 : Free diffusion walk
        fate = 6 : Thermal escape
        fate = 7 : Exciton becomes trapped
    """

    constants = np.zeros(7)
    constants[:4] = kin_const[:4]
    constants[-1] = kin_const[-1]
    constants[4] = k_nothing_d(t_step, *kin_const[:2])
    constants[5] = k_nothing(t_step, *kin_const[1::2])

    # inital exciton is free in E11
    fate = 5
    trapped = 0

    # Initiate matrix to store exciton fate
    exciton_fate = np.zeros(len(constants)+1)

    # Inital position of the exciton and defects
    pos_exc_0 = create_exciton(CNT_length)
    defects = create_defects(CNT_length, n_defects)

    while fate > 3:

        # step if exciton is free
        if trapped == 0:
            pos_exc_1 = round(pos_exc_0 + (
                2 * D_exc_nm_per_s * t_step * 1e-12)**0.5)

        # check if exciton became trapped
        if trapped == 0:
            pathway = np.arange(pos_exc_0, pos_exc_1)
            if np.in1d(pathway, defects).any():
                # set exciton to position of first encountered trap
                pos_exc_1 = defects[np.in1d(defects, pathway)][0]
                trapped = 1
                fate = 7
                exciton_fate[fate] += 1

        # quenching of the exciton at tube end
        if pos_exc_1 >= CNT_length:
            fate = 3
            exciton_fate[fate] += 1
            break

        # fate of a trapped exciton
        if trapped == 1:
            # calculate probability for fate of trapped exciton
            p_fate = np.array([e * random.uniform(0, 1)
                               for e in constants[::2]])
            # Store result for highest probability
            fate = 2*p_fate.argmax()
            exciton_fate[fate] += 1

        # fate of freely diffusing exciton
        else:
            # calculate probability for fate of free exciton
            p_fate = np.array([e * random.uniform(0, 1)
                               for e in constants[1::2]])
            # Store result for highest probability
            fate = (p_fate.argmax() * 2 + 1)
            exciton_fate[fate] += 1

        # insurance that there won't be an endless loop
        if exciton_fate.sum() > 1e6:
            print('Simulation exceeds 1e6 steps, loop aborded')
            return exciton_fate

        # set position to new starting position
        pos_exc_0 = pos_exc_1
        if fate == 6:
            pos_exc_0 += 2

    return exciton_fate


def exciton_simulation(t_step, kin_const, n_defects=10, CNT_length=L_nm,
                       r_exc_nm=R_nm):
    """
    Parameters
    ----------
    t_step : float
        Timestep in ps.
    constants : 1D array
        kinetic constants in order of:
        [k_d_r_per_s, k_r_per_s, k_d_nr_per_s, k_nr_per_s, k_dt_per_s]
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
        fate = 0 : E11* radiative decay
        fate = 1 : E11 radiative decay
        fate = 2 : E11* non-radiative decay
        fate = 3 : E11 non-radiative decay
        fate = 4 : Exciton stays in trap
        fate = 5 : Free diffusion walk
        fate = 6 : Thermal escape
        fate = 7 : Exciton becomes trapped
    """

    constants = np.zeros(7)
    constants[:4] = kin_const[:4]
    constants[-1] = kin_const[-1]
    constants[4] = k_nothing_d(t_step, *kin_const[:2])
    constants[5] = k_nothing(t_step, *kin_const[1::2])

    # inital exciton is free in E11
    fate = 4
    trapped = 0

    # Initiate matrix to store exciton fate
    exciton_fate = np.zeros(len(constants)+1)

    # Inital position of the exciton and defects
    pos_exc_0 = create_exciton(CNT_length)
    defects = create_defects(CNT_length, n_defects)

    # Masks defects which are too close together and result in non-radiative
    # decay
    defects = np.sort(defects)
    mask = [defects[1]-defects[0] >= r_exc_nm]
    mask.extend([True if defects[i+1]-defects[i] >= r_exc_nm
                 and defects[i]-defects[i-1] >= r_exc_nm
                 else False for i in np.arange(1, len(defects)-1)])
    mask.extend([defects[-1]-defects[-2] >= r_exc_nm])

    while fate > 3:

        # step if exciton is free
        if trapped == 0:
            pos_exc_1 = round(pos_exc_0 + (
                2 * D_exc_nm_per_s * t_step * 1e-12)**0.5)


        # check if exciton became trapped
        if trapped == 0:
            pathway = np.arange(pos_exc_0, pos_exc_1)
            if np.in1d(pathway, defects).any():
                # set exciton to position of first encountered trap
                pos_exc_1 = defects[np.in1d(defects, pathway)][0]
                # check if non-radiative decay takes plase
                if np.in1d(defects[mask], pos_exc_1).any():
                    trapped = 1
                    fate = 7
                    exciton_fate[fate] += 1
                else:
                    fate = 2
                    exciton_fate[fate] += 1
                    break

        # quenching of the exciton at tube end
        if pos_exc_1 >= CNT_length:
            fate = 2
            exciton_fate[fate] += 1
            break
                
        # fate of a trapped exciton
        if trapped == 1:
            # calculate probability for fate of trapped exciton
            p_fate = np.array([e * random.uniform(0, 1)
                               for e in constants[::2]])
            # Store result for highest probability
            fate = 2*p_fate.argmax()
            exciton_fate[fate] += 1
            if fate == 6:
                pos_exc_1 += r_exc_nm

        # fate of freely diffusing exciton
        else:
            # calculate probability for fate of free exciton
            p_fate = np.array([e * random.uniform(0, 1)
                               for e in constants[1::2]])
            # Store result for highest probability
            fate = (p_fate.argmax() * 2 + 1)
            exciton_fate[fate] += 1

        # insurance that there won't be an endless loop
        if exciton_fate.sum() > 1e6:
            print('Simulation exceeds 1e6 steps, loop aborded')
            return exciton_fate

        # set position to new starting position
        pos_exc_0 = pos_exc_1

    return exciton_fate