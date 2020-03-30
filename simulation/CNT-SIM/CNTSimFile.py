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
        self.calc_dict = None
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
        info = Text()
        display(info)

        self.calc_dict['n_photons'] = n_photons
        # initiate matrix size
        photons_fate = func(self.kin_const, **func_kwargs)
        info.value = f"Processing photon (1/ {n_photons})"

        # loop for the desired number of photons
        for p in np.arange(n_photons-1):
            photons_fate += func(**func_kwargs)
            info.value = f"Processing photon ({p+2}/ {n_photons})"

        # calculate the quantum yield
        quantum_yield = photons_fate[:2] / n_photons
        self.QY = quantum_yield
        return photons_fate, quantum_yield

    def defect_dependance(self, n_photons, func, n_defects, CNT_length,
                          t_step, **func_kwargs):
        self.calc_dict['n_defects'] = n_defects
        self.calc_dict['n_photons'] = n_photons
        self.calc_dict['CNT_length'] = CNT_length
        self.calc_dict['t_step'] = t_step
        self.QY = np.zeros((len(n_defects), 2))
        for i, n_def in enumerate(n_defects):
            _, self.QY[i, :] = self.photons_fate(n_photons, func,
                      {'n_defects': n_def, 'length': CNT_length,
                       't_step': t_step, **func_kwargs})

    def length_dependance(self, n_photons, func, CNT_length, defect_density,
                          t_step, **func_kwargs):
        self.calc_dict['CNT_length'] = CNT_length
        self.calc_dict['t_step'] = t_step
        self.calc_dict['n_photons'] = n_photons
        self.calc_dict['defect_density'] = defect_density
        self.n_defects = np.zeros(len(CNT_length))
        self.QY = np.zeros((len(self.n_defects), 2))
        for i, l_nm in enumerate(CNT_length):
            self.n_defects[i] = round(l_nm/defect_density)
            _, self.QY[i, :] = self.photons_fate(n_photons, func,
                      {'n_defects': self.n_defects[i], 'length': l_nm,
                       't_step': t_step, **func_kwargs})
        self.calc_dict['n_defects'] = self.n_defects
