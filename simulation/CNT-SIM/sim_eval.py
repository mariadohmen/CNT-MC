# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:24:55 2020

@author: Maria
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import re

from mpl_toolkits.axes_grid1 import make_axes_locatable

from .CNTSimFile import CNTSimFile


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def human_sorting(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [atof(c) for c in
            re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",
                       text)]


kin_const = []


def import_Sim_files(import_criterium, shape, kin_const_indeces):

    files = os.listdir('../sim_output/')
    files = [r'../sim_output/' + f
             for f in files if import_criterium
             in f if f.endswith('.h5')]
    files = sorted(files, key=human_sorting)
    CNT_sims = [CNTSimFile(i, kin_const) for i in files]
    n_def_calc, _ = CNT_sims[0].QY.shape

    # Initiate data cube
    CNT_cube = np.zeros(shape + (n_def_calc + 2), )

    # Load every QY in cube
    for i in np.arange(n_def_calc):
        CNT_cube[:, :, 2 * i] = np.array([sim.QY[i, 0]
                                          for sim in CNT_sims]).reshape(shape)
        CNT_cube[:, :, 2 * i + 1] = np.array(
                [sim.QY[i, 1] for sim in CNT_sims]).reshape(shape)
        CNT_cube[:, :, -2] = np.array(
                [sim.kin_const[kin_const_indeces[0]]
                 for sim in CNT_sims]).reshape(shape)
        CNT_cube[:, :, -1] = np.array(
                [sim.kin_const[kin_const_indeces[1]]
                 for sim in CNT_sims]).reshape(shape)

    return CNT_sims, CNT_cube




def colorbar(mappable, clabel=None):
    """Sets all parameter for the colorbar and sets it next to the
    mappable image. The colorbar is scaled so the AxesImage is not
    distorted.

    Parameters
    ----------
    mappable : AxesImage
        An AxesImage object as returned by plt.imshow for instance.

    Returns
    -------
    colorbar : Colorbar
        A colorbar at the righthandside of the mappable."""
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    if clabel is not None:
        return fig.colorbar(mappable, cax=cax, label=clabel)
    else:
        return fig.colorbar(mappable, cax=cax)


def plot_quantum_yield(data_cube, QY_index, title, xlabel='$k_{r}$',
                       ylabel='$k_{ed}$', decimal_points=2):
    """Plot a 2D image of QY variation when two rate constants are varied.

    Parameters
    ----------
    Data_cube: 3D Array
        3D data array, in the last two slices [:, :, -1] there need to be mesh
        grids of the to varied constants.
    QY_index: int
        Index of QY slice of the data cube with mappable to be plotted.
    title : str
        Title of the figure.
    xlabel : str, optional
        x axis label
    ylabel : str, optional
        y axis label
    decimal_points : int
        Number of decimal points for rounding the tick labels

    Returns
    --------
    fig : figure
        Figure in matplotlib"""

    fig, ax = plt.subplots(1, 1)

    img = ax.imshow(data_cube[:, :, QY_index]*100, extent=[-1, 1, -1, 1],
                    aspect='auto')
    I, J, K = data_cube.shape

    # set ticks
    ax.set_xticks(np.arange(-1 + 1 / J, 1, 2 / J))
    tick_list_1 = np.array([np.format_float_scientific(e, decimal_points)
                            for e in data_cube[0, :, -1]])
    tick_list_2 = np.array([np.format_float_scientific(e, decimal_points)
                            for e in data_cube[0, :, -2]])
    if len(np.unique(tick_list_1)) / len(tick_list_1) > len(
            np.unique(tick_list_2))/len(tick_list_2):
        ax.set_xticklabels(tick_list_1)
    else:
        ax.set_xticklabels(tick_list_2)

    ax.set_yticks(np.arange(-1 + 1 / I, 1, 2 / I))
    tick_list_1 = np.array([np.format_float_scientific(e, decimal_points)
                            for e in data_cube[:, 0, -1]])
    tick_list_2 = np.array([np.format_float_scientific(e, decimal_points)
                            for e in data_cube[:, 0, -2]])
    if len(np.unique(tick_list_2)) / len(tick_list_2) > len(
            np.unique(tick_list_1))/len(tick_list_1):
        ax.set_yticklabels(tick_list_2)
    else:
        ax.set_yticklabels(tick_list_1)

    # set labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_title(title)

    fig.colorbar(img)
    return fig
