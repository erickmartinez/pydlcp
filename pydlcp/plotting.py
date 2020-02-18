
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
from typing import List
from pydlcp import DLCPDataStore as dh5

axes_list = List[Axes]


class DataPlotter:

    _colorNorm: object = None
    _colorMap: mpl.cm = None
    _scalarMappable: mpl.colors.cm.ScalarMappable = None
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
    _h5DataStore: dh5.DLCPDataStore = None

    def __init__(self, data_store: dh5.DLCPDataStore):
        mpl.rcParams.update(self._defaultPlotStyle)
        self._h5DataStore = data_store

    def prepare_color_map(self, color_palette: str = 'winter'):
        # Find the number of nominal bias stored in the file
        npoints = self._h5DataStore.count_data_sets(path='/dlcp')
        biases = np.empty(npoints)
        for i in range(npoints):
            ds_name = '/dlcp/sweep_{0:d}'.format(i)
            metadata = self._h5DataStore.get_metadata(group=ds_name)
            biases = float(metadata['nominal_bias'])
            # Create a color map
        self._colorNorm = mpl.colors.Normalize(vmin=np.amin(biases), vmax=np.amax(biases))
        self._colorMap = mpl.cm.get_cmap(color_palette)
        # create a ScalarMappable and initialize a data structure
        self._scalarMappable = mpl.cm.ScalarMappable(cmap=self._timeColorMap, norm=self._colorNorm)

    def plo






