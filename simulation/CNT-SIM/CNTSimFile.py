# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:54:43 2020

@author: Maria
"""

import h5py
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn

from pathlib import Path
from ipywidgets.widgets import Text


class CNTSimFile:
    """Class which allows MC simulation of a random exciton walk"""

    def __init__(self, filepath, rate_const):
        self.filepath = filepath
        self.rate_const = rate_const
        self.QY = None
        self.p_fate = None
        self.calc_dict = {'date': datetime.datetime.now().strftime("%Y-%m-%d")}
        self.notebook_output = False
        if Path(filepath).is_file():
            self.load()
            warn("File already exists, kinetic constants ingored.")
            print('Existing file loaded successfully.')

    def __str__(self):
        return self.filepath

    def __repr__(self):
        return self.rate_const

    def load(self):
        """Loads existing hdf5 file found at the filepath.
        The File is expected to have the folloing keys:
        positions_x, positions_y, spectra and meta"""
        data = h5py.File(self.filepath)
        self.rate_const = data['rate_const'][:]
        self.QY = data['QY'][:]

        try:
            self.p_fate = data['p_fate'][:]
        except KeyError:
            warn('no p_fate key in file')

        try:
            self.QY_delta = data['QY_delta'][:]
        except KeyError:
            pass
        try:
            self.QY_ref = data['QY_ref'][:]
        except KeyError:
            pass
        try:
            self.p_fate_ref = data['p_fate_ref'][:]
        except KeyError:
            pass

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
            hdf5_file.create_dataset('rate_const', compression='gzip',
                                     data=self.rate_const, dtype=np.float32)
            hdf5_file.create_dataset('QY', compression='gzip',
                                     data=self.QY,
                                     dtype=np.float32)
            hdf5_file.create_dataset('p_fate', compression='gzip',
                                     data=self.p_fate,
                                     dtype=np.float32)
            try:
                hdf5_file.create_dataset('QY_delta', compression='gzip',
                                     data=self.QY_delta,
                                     dtype=np.float32)
                hdf5_file.create_dataset('QY_ref', compression='gzip',
                                     data=self.QY_ref,
                                     dtype=np.float32)
                hdf5_file.create_dataset('p_fate_ref', compression='gzip',
                                     data=self.p_fate_ref,
                                     dtype=np.float32)
            except AttributeError:
                pass
            grp = hdf5_file.create_group('dict', )
            for key, value in self.calc_dict.items():
                grp.attrs[key] = value
        except:
            print('Datasets could not be created')
        finally:
            hdf5_file.close()

    def photons_fate(self, n_photons, func, rate_const, func_kwargs={}):
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

        print('Start of exiton simulation:', datetime.datetime.now())
        start_p = time.time()

        self.calc_dict['n_photons'] = n_photons
        # initiate matrix size
        photons_fate = func(rate_const=rate_const, **func_kwargs)
        if self.notebook_output is True:
            info = Text()
            display(info)
            info.value = f"Processing photon (1/ {n_photons})"

        # loop for the desired number of photons
        for p in np.arange(n_photons-1):
            photons_fate += func(rate_const=rate_const, **func_kwargs)
            if self.notebook_output is True:
                info.value = f"Processing photon ({p+2}/ {n_photons})"

        # calculate the quantum yield
        quantum_yield = photons_fate[:2] / n_photons

        end = time.time()
        elapsed = end - start_p
        print('End exiton simulation:', datetime.datetime.now())
        print('Elapsed time:', time.strftime("%H:%M:%S", time.gmtime(elapsed)))
        return photons_fate, quantum_yield

    def defect_dependence(self, n_photons, func, n_defects, func_kwargs={},
                          plot=False):
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

        Yields
        ------
        QY : 2D array
            Array of quantum yields at different defect density in
            QY object.
        calc_dict : dict
            Updates information in the calc_dict object.
        """
        print('start of calculation:', datetime.datetime.now())
        start = time.time()

        self.calc_dict['n_defects'] = n_defects
        self.calc_dict['function'] = func.__name__
        self.calc_dict['method'] = self.defect_dependence.__name__
        self.calc_dict = {**self.calc_dict, **func_kwargs}

        self.QY = np.zeros((len(n_defects), 2))
        p_fate, _ = self.photons_fate(1, func, self.rate_const,
                                      {'n_defects': 0, **func_kwargs})
        self.p_fate = np.zeros((len(n_defects), np.size(p_fate)))

        for i, n_def in enumerate(n_defects):
            print(f'defect numbers processed(({i}/ {len(n_defects)}))')
            self.p_fate[i, :], self.QY[i, :] = self.photons_fate(
                    n_photons, func, self.rate_const,
                    {'n_defects': n_def, **func_kwargs})

        end = time.time()
        elapsed = end - start
        print(datetime.datetime.now())
        print('elapsed time:', time.strftime("%H:%M:%S", time.gmtime(elapsed)))
        if plot is True:
            fig = plt.subplot()
            plt.plot(self.calc_dict['n_defects'], self.QY[:, 0] * 100,
                     label='E11')
            plt.plot(self.calc_dict['n_defects'], self.QY[:, 1] * 100,
                     label='E11*')
            plt.xlabel('number of defects')
            plt.ylabel('quantum yield / %')
            plt.title('Defect dependence, l = {} nm'.format(
                      self.calc_dict['CNT_length']))
            return fig

    def length_dependence(self, n_photons, func, CNT_length, defect_density,
                          func_kwargs={}, plot=False):
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

        Yields
        ------
        QY : 2D array
            Array of quantum yields at different defect density in
            QY object.
        calc_dict : dict
            Updates information in the calc_dict object.
        """
        print('Start of calculation:', datetime.datetime.now())
        start = time.time()

        self.calc_dict['CNT_length'] = CNT_length
        self.calc_dict['defect_density'] = defect_density
        self.calc_dict['function'] = func.__name__
        self.calc_dict['method'] = self.length_dependence.__name__
        self.calc_dict = {**self.calc_dict, **func_kwargs}

        self.n_defects = np.zeros(len(CNT_length))
        self.QY = np.zeros((len(self.n_defects), 2))
        p_fate, _ = self.photons_fate(1, func, self.rate_const,
                                      {'n_defects': 0, 'CNT_length': 750,
                                       **func_kwargs})
        self.p_fate = np.zeros((len(self.n_defects), np.size(p_fate)))

        for i, l_nm in enumerate(CNT_length):
            print(f'CNT lengths processed(({i}/ {len(CNT_length)}))')
            self.n_defects[i] = round(l_nm/defect_density)
            self.p_fate[i, :], self.QY[i, :] = self.photons_fate(
                    n_photons, func, self.rate_const,
                    {'n_defects': int(self.n_defects[i]), 'CNT_length': l_nm,
                     **func_kwargs})
        self.calc_dict['n_defects'] = self.n_defects

        end = time.time()
        elapsed = end - start
        print('End exiton simulation:', datetime.datetime.now())
        print('Elapsed time:', time.strftime("%H:%M:%S", time.gmtime(elapsed)))
        if plot is True:
            fig = plt.subplot()
            plt.plot(self.calc_dict['CNT_length'], self.QY[:, 0] * 100,
                     label='E11')
            plt.plot(self.calc_dict['CNT_length'], self.QY[:, 1] * 100,
                     label='E11*')
            plt.xlabel('length of the nanotube')
            plt.ylabel('quantum yield / %')
            plt.title('lenght dependence, defect distance = {} nm'.format(
                      self.calc_dict['defect_density']))
            return fig

    def diffusion_dependence(self, n_photons, func, diff_const, func_kwargs={},
                             plot=False):
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
        diff_const : array
            2D numpy array [:, 0] is for diffusion constant of the exited
            state and [:, 1] contains the diffusion constants for the dark
            state

        Yields
        ------
        QY : 2D array
            Array of quantum yields at different defect density in
            QY object.
        calc_dict : dict
            Updates information in the calc_dict object.
        """
        print('Start of calculation:', datetime.datetime.now())
        start = time.time()

        self.calc_dict['Diff_const'] = diff_const
        self.calc_dict['function'] = func.__name__
        self.calc_dict['method'] = self.diffusion_dependence.__name__
        self.calc_dict = {**self.calc_dict, **func_kwargs}

        self.QY = np.zeros((self.calc_dict['Diff_const'].shape[0], 2))

        p_fate, _ = self.photons_fate(1, func, self.rate_const,
                                      {**func_kwargs})
        self.p_fate = np.zeros((self.calc_dict['Diff_const'].shape[0],
                                np.size(p_fate)))

        for i in np.arange(diff_const.shape[0]):
            print(f'diff constant set processed({i}/ {diff_const.shape[0]})')

            self.p_fate[i, :], self.QY[i, :] = self.photons_fate(
                    n_photons, func, self.rate_const,
                    {'Diff_exc_e': diff_const[i, 0],
                     'Diff_exc_d': diff_const[i, 1],
                     **func_kwargs})
        end = time.time()
        elapsed = end - start
        print('End exiton simulation:', datetime.datetime.now())
        print('Elapsed time:', time.strftime("%H:%M:%S", time.gmtime(elapsed)))
        if plot is True:
            fig, axes = plt.subplots(nrows=1, ncols=2)
            axes[0].plot(self.calc_dict['Diff_const'][:, 0],
                         self.QY[:, 0] * 100, label='E11*')
            axes[0].plot(self.calc_dict['Diff_const'][:, 0],
                         self.QY[:, 1] * 100, label='E11')
            axes[0].set_xlabel('diff_exc_e / nm s$^{-1}$')
            axes[0].set_ylabel('quantum yield / %')
            axes[0].set_title('n_def = {} nm, l = {} nm'.format(
                      self.calc_dict['n_defects'],
                      self.calc_dict['CNT_length']))
            axes[1].plot(self.calc_dict['Diff_const'][:, 1],
                         self.QY[:, 0] * 100, label='E11*')
            axes[1].plot(self.calc_dict['Diff_const'][:, 1],
                         self.QY[:, 1] * 100, label='E11')
            axes[1].set_xlabel('diff_exc_d / nm s$^{-1}$')
            axes[1].set_ylabel('quantum yield / %')
            axes[1].set_title('n_def = {} nm, l = {} nm'.format(
                      self.calc_dict['n_defects'],
                      self.calc_dict['CNT_length']))
            plt.legend()
            plt.tight_layout()
            return fig

    def referenced_diffusion_dependence(self, n_photons, func, diff_const,
                                        ref_diff_const, func_kwargs={},
                                        plot=False):
        """Calculates the realtive difference of a variation of the diffusion
        constants to a reference diffusion constant.

        Parameters
        ----------
        ref_diff_const : array
            array like with two positions, 0 is the diffusion constant of the
            exited state and 1 the diffusion constant of the dark state.

        Others arguments see diffusion_dependence.
        """
        self.calc_dict['ref_diff_const'] = ref_diff_const
        self.calc_dict['function'] = func.__name__
        self.calc_dict['method'] = self.referenced_diffusion_dependence.__name__

        self.diffusion_dependence(n_photons, func, diff_const,
                                  {**func_kwargs})
        self.p_fate_ref, self.QY_ref = self.photons_fate(
                                            n_photons, func, self.rate_const,
                                            {'Diff_exc_e': ref_diff_const[0],
                                             'Diff_exc_d': ref_diff_const[1],
                                             **func_kwargs})
        self.QY_delta = (self.QY - self.QY_ref) / self.QY_ref

    def rate_const_dependence(self, n_photons, func, constant_dependence,
                             chosen_const, constant_names, func_kwargs={},
                              plot=False):
        """
        Calculates Quantum Yield for a varity of rate constants and calculates
        the difference to the reference set given upon initiation of the
        CNTSimFile

        Parameters
        ----------
        constant_dependence : array
            2D array of shape (n, m), with n variations of m rate constants.
        chosen_const : list
            Orderd list of all the names of m rate constants given in
            constant_array as stings.
        constant_names : list
            Ordered list of all the names of rate constants required for the
            function as strings.

        Yields
        ------
        constant_array : array
            2D array with full set of rate constants used for each simulation
            step.
        """
        print('Start of calculation:', datetime.datetime.now())
        start = time.time()

        shape_nm = constant_dependence.shape
        N = shape_nm[0]
        try:
            M = shape_nm[1]
        except IndexError:
            M = 1

        self.calc_dict['constant_dependence'] = constant_dependence
        self.calc_dict['chosen_const'] = chosen_const
        self.calc_dict['constant_names'] = constant_names
        self.calc_dict['function'] = func.__name__
        self.calc_dict['method'] = self.rate_const_dependence.__name__
        self.calc_dict = {**self.calc_dict, **func_kwargs}

        # Caluclate reference for rate constants without dopamine addition.
        self.p_fate_ref, self.QY_ref = self.photons_fate(
                    n_photons, func, self.rate_const, {**func_kwargs})

        #Check if all names where given
        if len(constant_names) != len(self.rate_const):
            raise ValueError('Number of rate constants given upon initiation '
                             'does not match the number of names given.')
            return

        # check if all rate constants are varried
        if M == len(self.rate_const):
            self.calc_dict['constant_array'] = constant_dependence
            constant_array = constant_dependence
        else:

            name_loc = [True if i in chosen_const else False for i in
                        constant_names]
            name_idx = np.where(name_loc)[0]
            constant_array = np.zeros((N, len(self.rate_const)))
            for n in np.arange(N):
                constant_array[n, :] = self.rate_const
                for m in np.arange(M):
                    constant_array[n, name_idx[m]] = constant_dependence[n, m]
            self.calc_dict['constant_array'] = constant_array

        self.QY = np.zeros((N, 2))
        self.p_fate = np.zeros((N, np.size(self.p_fate_ref)))
        for i in np.arange(N):
            print(f'rate constant set processed({i+1}/ {N})')
            self.p_fate[i, :], self.QY[i, :] = self.photons_fate(
                    n_photons, func, constant_array[i, :],
                    {**func_kwargs})
        self.QY_delta = (
            self.QY - self.QY_ref) / self.QY_ref
        end = time.time()
        elapsed = end - start
        print('End exiton simulation:', datetime.datetime.now())
        print('Elapsed time:', time.strftime("%H:%M:%S", time.gmtime(elapsed)))

    def parameter_dependence(self, n_photons, func, constant_dependence,
                             chosen_const, constant_names, diff_const,
                             ref_diff_const, func_kwargs={},
                              plot=False):
        """
        Calculates Quantum Yield for a varity of rate constants and diffusion
        constant. calculates the difference to the reference set given upon
        initiation of the CNTSimFile.

        Parameters
        ----------
        constant_dependence : array
            2D array of shape (n, m), with n variations of m rate constants.
        chosen_const : list
            Orderd list of all the names of m rate constants given in
            constant_array as stings.
        constant_names : list
            Ordered list of all the names of rate constants required for the
            function as strings.
        func : callable
            Function which returns the quantum yield. Also needs to take
            the following arguments:
        Diff_exc_e : float
            Diffusion constant for excited exciton, global constant as default
        Diff_exc_d : float
            Diffusion constant for dark exciton, global constant as default
        diff_const : 2D array
            2D numpy array [:, 0] is for diffusion constant of the exited
            state and [:, 1] contains the diffusion constants for the dark
            state.
        ref_diff_const : array
            array like with two positions, 0 is the diffusion constant of the
            exited state and 1 the diffusion constant of the dark state.

        Yields
        ------
        constant_array : array
            2D array with full set of rate constants used for each simulation
            step.
        """
        print('Start of calculation:', datetime.datetime.now())
        start = time.time()

        shape_nm = constant_dependence.shape
        N = shape_nm[0]
        try:
            M = shape_nm[1]
        except IndexError:
            M = 1
        self.calc_dict['Diff_const'] = diff_const
        self.calc_dict['ref_diff_const'] = ref_diff_const
        self.calc_dict['constant_dependence'] = constant_dependence
        self.calc_dict['chosen_const'] = chosen_const
        self.calc_dict['constant_names'] = constant_names
        self.calc_dict['function'] = func.__name__
        self.calc_dict['method'] = self.parameter_dependence.__name__
        self.calc_dict = {**self.calc_dict, **func_kwargs}

        # Caluclate reference for rate constants without dopamine addition.
        self.p_fate_ref, self.QY_ref = self.photons_fate(
                    n_photons, func, self.rate_const,
                    {'Diff_exc_e': ref_diff_const[0],
                     'Diff_exc_d': ref_diff_const[1],
                     **func_kwargs})

        #Check if all names where given
        if len(constant_names) != len(self.rate_const):
            raise ValueError('Number of rate constants given upon initiation '
                             'does not match the number of names given.')
            return

        # check if all rate constants are varried
        if M == len(self.rate_const):
            self.calc_dict['constant_array'] = constant_dependence
            constant_array = constant_dependence
        else:

            name_loc = [True if i in chosen_const else False for i in
                        constant_names]
            name_idx = np.where(name_loc)[0]
            constant_array = np.zeros((N, len(self.rate_const)))
            for n in np.arange(N):
                constant_array[n, :] = self.rate_const
                for m in np.arange(M):
                    constant_array[n, name_idx[m]] = constant_dependence[n, m]
            self.calc_dict['constant_array'] = constant_array

        # Check if the number of Diff constants are given is different
        if N != diff_const.shape[0]:
            warn('the number of diffusion constant variations not matichng '
                 'the number of rate constant variations. N_Diff adjusted')
            diff_const[:, 0] = np.linspace(diff_const[0, 0], diff_const[-1, 0],
                                           N)
            diff_const[:, 1] = np.linspace(diff_const[0, 1], diff_const[-1, 1],
                                           N)

        self.QY = np.zeros((N, 2))
        self.p_fate = np.zeros((N, np.size(self.p_fate_ref)))
        for i in np.arange(N):
            print(f'rate constant set processed({i+1}/ {N})')
            self.p_fate[i, :], self.QY[i, :] = self.photons_fate(
                    n_photons, func, constant_array[i, :],
                    {'Diff_exc_e': diff_const[i, 0],
                     'Diff_exc_d': diff_const[i, 1],
                     **func_kwargs})
        self.QY_delta = (self.QY - self.QY_ref) / self.QY_ref
        end = time.time()
        elapsed = end - start
        print('End exiton simulation:', datetime.datetime.now())
        print('Elapsed time:', time.strftime("%H:%M:%S", time.gmtime(elapsed)))
