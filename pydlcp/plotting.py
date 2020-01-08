"""
This class provides plotting methods to display CV and DLCP graphs

@author: Erick R Martinez Loran <erickrmartinez@gmail.com>
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
from typing import List
from pydlcp import datastorage, bts

h5_list = List[datastorage.H5Store]
axes_list = List[Axes]


class DataPlotter:

    _bts: bts.BTS = None
    _defaultPlotStyle = {'font.size': 16,
                         'font.family': 'Arial',
                         'font.weight': 'regular',
                         'legend.fontsize': 16,
                         'mathtext.fontset': 'custom',
                         'mathtext.rm': 'Times New Roman',
                         'mathtext.it': 'Times New Roman:italic',
                         'mathtext.cal': 'Times New Roman:italic',
                         'mathtext.bf': 'Times New Roman:bold',
                         'xtick.direction': 'in',
                         'ytick.direction': 'in',
                         'xtick.major.size': 4.5,
                         'xtick.major.width': 1.75,
                         'ytick.major.size': 4.5,
                         'ytick.major.width': 1.75,
                         'xtick.minor.size': 2.75,
                         'xtick.minor.width': 1.0,
                         'ytick.minor.size': 2.75,
                         'ytick.minor.width': 1.0,
                         'ytick.right': False,
                         'lines.linewidth': 3,
                         'lines.markersize': 10,
                         'lines.markeredgewidth': 1.0,
                         'axes.labelpad': 6.0,
                         'axes.labelsize': 20,
                         'axes.labelweight': 'regular',
                         'axes.linewidth': 2,
                         'axes.titlesize': 20,
                         'axes.titleweight': 'bold',
                         'axes.titlepad': 7,
                         'figure.titleweight': 'bold',
                         'figure.dpi': 100}
    _h5DataStores: h5_list = []

    def __init__(self, bts_experiment: bts.BTS):
        self._bts = bts_experiment
        mpl.rcParams.update(self._defaultPlotStyle)
        # Create a color map
        self._colorNorm = mpl.colors.Normalize(vmin=0, vmax=bts_experiment.max_time)
        self._timeColorMap = mpl.cm.get_cmap('rainbow')
        # create a ScalarMappable and initialize a data structure
        self._scalarMappable = mpl.cm.ScalarMappable(cmap=self._timeColorMap, norm=self._colorNorm)

    def plot_cv(self, fig: Figure, axes: axes_list,  clean: bool = True):
        for i, d in enumerate(self._bts.clean_devices):
            h5_data_store = self._bts.get_device_storage(device=d, clean=clean)
            cv_data = h5_data_store.get_bts_cv()
            stress_time = cv_data['stress_time']
            ax: Axes = axes[i]
            # if the axis already has lines just append
            n_line = len(ax.lines)
            for j, t in enumerate(stress_time[n_line:]):
                c = self._timeColorMap(self._colorNorm(t))
                ax.plot(cv_data['voltage'][j], cv_data['capacitace'][j], color=c)






