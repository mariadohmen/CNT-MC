# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:54:43 2020

@author: Maria
"""

import h5py
import numpy as np

from pathlib import Path
from ipywidgets.widgets import Text


class CNTSimFile:
    """Class which allows MC simulation of a random exciton walk"""
    def __init__(self, filepath, kin_const):
        self.filepath = filepath
        self.kin_const = kin_const
        self.QY = None
        self.n_photons = None
        self.CNT_length = None
        self.t_step = None
        self.n_defects = None
        if Path(filepath).is_file():
            self.load()
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
        self.n_photons = data['n_photons'][:]
        self.CNT_length = data['CNT_length'][:]
        self.t_step = data['t_step'][:]
        self.n_defects = data['n_defects'][:]
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
            hdf5_file.create_dataset('n_photons', compression='gzip',
                                     data=self.n_photons,
                                     dtype=np.float32)
            hdf5_file.create_dataset('CNT_length', compression='gzip',
                                     data=self.CNT_length,
                                     dtype=np.float32)
            hdf5_file.create_dataset('t_step', compression='gzip',
                                     data=self.t_step,
                                     dtype=np.float32)
            hdf5_file.create_dataset('n_defects', compression='gzip',
                                     data=self.n_defects,
                                     dtype=np.float32)
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
        info = Text()
        display(info)

        self.n_photons = n_photons
        # initiate matrix size
        photons_fate = func(self.kin_const, **func_kwargs)
        info.value = f"Processing photon (1/ {n_photons})"

        # loop for the desired number of photons
        for p in np.arange(self.n_photons-1):
            photons_fate += func(**func_kwargs)
            info.value = f"Processing photon ({p+2}/ {n_photons})"

        # calculate the quantum yield
        quantum_yield = photons_fate[:2] / n_photons
        self.QY = quantum_yield
        return photons_fate, quantum_yield
