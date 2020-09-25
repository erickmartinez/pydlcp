# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:31:40 2020

@author: Erick
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
import h5py
import platform
import os

data_folder = r'G:\My Drive\Research\PVRD1\DATA\DLCP\SiOx\60C_0.3MVcm_D69'
filename_lf = 'D69_clean_low_freq_20200309.h5'
filename_hf = 'D69_clean_high_freq_20200224.h5'

files = [filename_lf, filename_hf]
#files = [filename_hf]
freqs = ['20 kHz Traps+dopants', '1 MHz dopants only']
#freqs = ['1 MHz']
linespec = ['o', 's']
colors = ['C0','C1']

color_palette = 'winter'

dlcp_type = np.dtype([('osc_level', 'd'), ('bias', 'd'), ('nominal_bias', 'd'), ('V', 'd'), ('C', 'd')])
cfit_type = np.dtype([('osc_level', 'd'), ('C_fit', 'd'), ('lpb', 'd'), ('upb', 'd')])
ndl_type = np.dtype([('xvar', 'd'), ('xvar_err', 'd'), ('NDL', 'd'), ('NDL_err', 'd')])


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


defaultPlotStyle = {'font.size': 18,
                    'font.family': 'Arial',
                    'font.weight': 'regular',
                    'legend.fontsize': 18,
                    'mathtext.fontset': 'custom',
                    'mathtext.rm': 'Times New Roman',
                    'mathtext.it': 'Times New Roman:italic',  # 'Arial:italic',
                    'mathtext.cal': 'Times New Roman:italic',  # 'Arial:italic',
                    'mathtext.bf': 'Times New Roman:bold',  # 'Arial:bold',
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
                    'lines.linewidth': 2.5,
                    'lines.markersize': 10,
                    'lines.markeredgewidth': 0.85,
                    'axes.labelpad': 5.0,
                    'axes.labelsize': 20,
                    'axes.labelweight': 'regular',
                    'axes.linewidth': 1.25,
                    'axes.titlesize': 18,
                    'axes.titleweight': 'bold',
                    'axes.titlepad': 6,
                    'figure.titleweight': 'bold',
                    'figure.dpi': 100}

if __name__ == '__main__':
    mpl.rcParams.update(defaultPlotStyle)
    if platform.system() == 'Windows':
        data_folder = r'\\?\\' + data_folder
    
    fig = plt.figure()
    fig.set_size_inches(5.5, 4.5, forward=True)
#    fig.subplots_adjust(hspace=0.15, wspace=0.5)
    gs0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, width_ratios=[1])
    ax1 = fig.add_subplot(gs0[0, 0])
        
    for i,fn in enumerate(files):
        data_file = os.path.join(data_folder, fn)
        with h5py.File(data_file, 'r') as hf:
            print('Opening data file: \'{0}\'.'.format(fn))
            dlcp_ds = hf['dlcp']
            nb_start = float(dlcp_ds.attrs['nominal_bias_start'])
            nb_step = float(dlcp_ds.attrs['nominal_bias_step'])
            nb_stop = float(dlcp_ds.attrs['nominal_bias_stop'])
            nominal_biases = np.arange(start=nb_start, stop=nb_stop + nb_step,
                                       step=nb_step)
            
            idxs = nominal_biases >= -5.0
            ndl_data = np.array(hf['NDL'], dtype=ndl_type)
            xvar = ndl_data['xvar']
            ndl = ndl_data['NDL']
            ndl_err = ndl_data['NDL_err']
            
            ndl = ndl[idxs]
            xvar = xvar[idxs]
            ndl_err = ndl_err[idxs]
    
            ax1.errorbar(xvar*1E7, ndl, yerr=ndl_err, 
                         capsize=2.0, capthick=1.0, fillstyle='none',
                         marker=linespec[i], mew=1.25, label=freqs[i], 
                         linestyle='none')
    ax1.set_yscale('log')
    
    ax1.set_ylim([5E13,1E15])
    ax1.yaxis.set_ticks_position('both')
    
    
    locmaj = mpl.ticker.LogLocator(base=10.0,numticks=3) 
    locmin = mpl.ticker.LogLocator(base=10.0,numticks=11, subs=np.arange(0.1,1.0,0.1)) 
    
    ax1.yaxis.set_major_locator(locmaj)
    ax1.yaxis.set_minor_locator(locmin)
            
            
    ax1.set_xlabel(r'$\epsilon A/C$ (nm)')
    ax1.set_ylabel(r'N$_{\mathregular{DL}}$ (cm$^{-3}$)')

    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((-3, 3))
    ax1.xaxis.set_major_formatter(xfmt)

    
    leg = ax1.legend(loc='upper left',frameon=False)
    for i, text in enumerate(leg.get_texts()):
        text.set_color(colors[i])
    

    ax1.yaxis.set_major_locator(locmaj)
    ax1.yaxis.set_minor_locator(locmin)

    ax1.xaxis.set_major_locator(mticker.MaxNLocator(5, prune=None))
    ax1.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))

    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.show()
    
    

    filetag = '__'.join([os.path.splitext(os.path.basename(f))[0] for f in files])

    fig.savefig(os.path.join(data_folder, '{}.png'.format(filetag)), dpi=600)
    fig.savefig(os.path.join(data_folder, '{}.eps'.format(filetag)), dpi=600,
                format='eps')
        
        

