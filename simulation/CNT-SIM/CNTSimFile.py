# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:54:43 2020

@author: Maria
"""

import h5py
import datetime
import time
import numpy as np
from warnings import warn

from pathlib import Path
from ipywidgets.widgets import Text


class CNTSimFile:
    """Class which allows MC simulation of a random exciton walk"""
    
    def __init__(self, filepath, kin_const):
        self.filepath = filepath
        self.kin_const = kin_const
        self.QY = None
        self.p_fate = None
        self.calc_dict = dict()
        self.n_defects = None
        self.notebook_output = False
        if Path(filepath).is_file():
            self.load()
            warn("File already exists, kinetic constants ingored.")
            print('Existing file loaded successfully.')

    def __str__(self):
        return self.filepath

    def __repr__(self):
        return self.kin_const

    def load(self):
        """Loads existing hdf5 file found at the filepath.
        The File is expected to have the folloing keys:
        positions_x, positions_y, spectra and meta"""
        data = h5py.File(self.filepath)
        self.kin_const = data['kin_const'][:]
        self.QY = data['QY'][:]
        try:
            self.p_fate = data['p_fate'][:]
        except KeyError:
            warn('no p_fate key in file')
        self.calc_dict = {k: v for k, v in data['dict'].attrs.items()}
        data.close()

    def save(self):
        """Save the CNTSimFile in hd5f format at the given filepath."""
        try:
            hdf5_file = h5py.File(self.filepath, 'w')
        except IOError:
            print('File could not be opened')
            hdf5_file.close()
        try:
            hdf5_file.create_dataset('kin_const', compression='gzip',
                                     data=self.kin_const, dtype=np.float32)
            hdf5_file.create_dataset('QY', compression='gzip',
                                     data=self.QY,
                                     dtype=np.float32)
            hdf5_file.create_dataset('p_fate', compression='gzip',
                                     data=self.p_fate,
                                     dtype=np.float32)
            grp = hdf5_file.create_group('dict', )
            for key, value in self.calc_dict.items():
                grp.attrs[key] = value
        except:
            print('Datasets could not be created')
        finally:
            hdf5_file.close()

    def photons_fate(self, n_photons, func, func_kwargs={}):
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
            Quantum yield of the E11 and E11* radiative decay."""

        print('start of exiton simulation:', datetime.datetime.now())
        start_p = time.time()

        self.calc_dict['n_photons'] = n_photons
        # initiate matrix size
        photons_fate = func(kin_const=self.kin_const, **func_kwargs)
        if self.notebook_output is True:
            info = Text()
            display(info)
            info.value = f"Processing photon (1/ {n_photons})"

        # loop for the desired number of photons
        for p in np.arange(n_photons-1):
            photons_fate += func(kin_const=self.kin_const, **func_kwargs)
            if self.notebook_output is True:
                info.value = f"Processing photon ({p+2}/ {n_photons})"

        # calculate the quantum yield
        quantum_yield = photons_fate[:2] / n_photons

        end = time.time()
        elapsed = end - start_p
        print(datetime.datetime.now())
        print('elapsed time:', time.strftime("%H:%M:%S", time.gmtime(elapsed)))
        return photons_fate, quantum_yield

    def defect_dependence(self, n_photons, func, n_defects, func_kwargs={}):
        """
        Calculates the dependance of the quantum yield on the number of
        defects.

        Parameters
        ----------
        n_photons : int
            Number of photons used to calculate a single QY value.
        func : callable
            Function which returns the quantum yield. Also needs to take
            the following arguments:
        n_defects : list
            List of integers for the varied number of defects on the
            nanotube

        Returns
        -------
        QY : 2D array
            Array of quantum yields at different defect density in
            QY object.
        calc_dict : dict
            Updates information in the calc_dict object.
        """
        print('start of calculation:', datetime.datetime.now())
        start = time.time()

        self.calc_dict['n_defects'] = n_defects
        self.calc_dict.update(func_kwargs)
        self.calc_dict = {**self.calc_dict, **func_kwargs}

        self.QY = np.zeros((len(n_defects), 2))
        p_fate, _ = self.photons_fate(1, func, {'n_defects': 0, **func_kwargs})
        self.p_fate = np.zeros((len(n_defects), np.size(p_fate)))

        for i, n_def in enumerate(n_defects):
            print(f'exciton processed(({i}/ {len(n_defects)}))')
            self.p_fate[i, :], self.QY[i, :] = self.photons_fate(
                    n_photons, func, {'n_defects': n_def, **func_kwargs})

        end = time.time()
        elapsed = end - start
        print(datetime.datetime.now())
        print('elapsed time:', time.strftime("%H:%M:%S", time.gmtime(elapsed)))

    def length_dependence(self, n_photons, func, CNT_length, defect_density,
                          func_kwargs={}):
        """
        Calculates the dependence of the quantum yield on the number of
        defects.

        Parameters
        ----------
        n_photons : int
            Number of photons used to calculate a single QY value.
        func : callable
            Function which returns the quantum yield. Also needs to take
            the following arguments:
       CNT_length : list
            List of the varied length of the CNT in nm.
        defect_density : float
            Average between two defects in nm.

        Returns
        -------
        QY : 2D array
            Array of quantum yields at different defect density in
            QY object.
        calc_dict : dict
            Updates information in the calc_dict object.
        """
        print('start of calculation:', datetime.datetime.now())
        start = time.time()

        self.calc_dict['CNT_length'] = CNT_length
        self.calc_dict['defect_density'] = defect_density
        self.calc_dict = {**self.calc_dict, **func_kwargs}

        self.n_defects = np.zeros(len(CNT_length))
        self.QY = np.zeros((len(self.n_defects), 2))
        p_fate, _ = self.photons_fate(1, func, {'n_defects': 0,
                                                'CNT_length': 300,
                                                **func_kwargs})
        self.p_fate = np.zeros((len(self.n_defects), np.size(p_fate)))

        for i, l_nm in enumerate(CNT_length):
            print(f'exciton processed(({i}/ {len(CNT_length)}))')
            self.n_defects[i] = round(l_nm/defect_density)
            self.p_fate[i, :], self.QY[i, :] = self.photons_fate(
                    n_photons, func,
                    {'n_defects': int(self.n_defects[i]), 'CNT_length': l_nm,
                     **func_kwargs})
        self.calc_dict['n_defects'] = self.n_defects

        end = time.time()
        elapsed = end - start
        print(datetime.datetime.now())
        print('elapsed time:', time.strftime("%H:%M:%S", time.gmtime(elapsed)))

    def diffusion_dependence(self, n_photons, func,
                             diff_dependence, func_kwargs={}):
        """
        Calculates the dependance of the quantum yield on the number of
        defects.

        Parameters
        ----------
        n_photons : int
            Number of photons used to calculate a single QY value.
        func : callable
            Function which returns the quantum yield. Also needs to take
            the following arguments:
        Diff_exc_e : float
            Diffusion constant for excited exciton, global constant as default
        Diff_exc_d : float
            Diffusion constant for dark exciton, global constant as default
        diff_dependence : array
            2D numpy array [0, :] is for diffusion constant of the exited
            state and [1, :] contains the diffusion constants for the dark
            state

        Returns
        -------
        QY : 2D array
            Array of quantum yields at different defect density in
            QY object.
        calc_dict : dict
            Updates information in the calc_dict object.
        """
        print('To be implemented')
        print('start of calculation:', datetime.datetime.now())
        start = time.time()

        self.calc_dict = {**self.calc_dict, **func_kwargs}
        self.diff_speed = diff_dependence

        self.QY = np.zeros((self.diff_dependence.shape[1], 2))

        p_fate, _ = self.photons_fate(1, func, {**func_kwargs})
        self.p_fate = np.zeros((self.diff_dependence.shape[1],
                                np.size(p_fate)))

        for i in diff_dependence.shape[1]):
            print(f'exciton processed(({i}/ diff_dependence.shape[1]))')

            self.p_fate[i, :], self.QY[i, :] = self.photons_fate(
                    n_photons, func,
                    {Diff_exc_e=diff_dependence[0, i],
                     Diff_exc_d=diff_dependence[1, i]
                     **func_kwargs})
        end = time.time()
        elapsed = end - start
        print(datetime.datetime.now())
        print('elapsed time:', time.strftime("%H:%M:%S", time.gmtime(elapsed)))

        