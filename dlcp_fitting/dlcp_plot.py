# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:54:51 2020

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
filename = 'D69_clean_low_freq_20200309.h5'

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


defaultPlotStyle = {'font.size': 14,
                    'font.family': 'Arial',
                    'font.weight': 'regular',
                    'legend.fontsize': 14,
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
                    'axes.labelsize': 16,
                    'axes.labelweight': 'regular',
                    'axes.linewidth': 1.25,
                    'axes.titlesize': 16,
                    'axes.titleweight': 'bold',
                    'axes.titlepad': 6,
                    'figure.titleweight': 'bold',
                    'figure.dpi': 100}

if __name__ == '__main__':
    if platform.system() == 'Windows':
        data_folder = r'\\?\\' + data_folder

    data_file = os.path.join(data_folder, filename)
    mpl.rcParams.update(defaultPlotStyle)

    # Get the drive levels
    with h5py.File(data_file, 'r') as hf:
        dlcp_ds = hf['dlcp']
        nb_start = float(dlcp_ds.attrs['nominal_bias_start'])
        nb_step = float(dlcp_ds.attrs['nominal_bias_step'])
        nb_stop = float(dlcp_ds.attrs['nominal_bias_stop'])
        nominal_biases = np.arange(start=nb_start, stop=nb_stop + nb_step,
                                   step=nb_step)

        normalize = mpl.colors.Normalize(vmin=np.amin(nominal_biases),
                                         vmax=np.amax(nominal_biases))

        cm = mpl.cm.get_cmap(color_palette)
        nb_colors = [cm(normalize(bb)) for bb in nominal_biases]
        scalar_maps = mpl.cm.ScalarMappable(cmap=cm, norm=normalize)

        fig = plt.figure()
        fig.set_size_inches(6.5, 3.0, forward=True)
        fig.subplots_adjust(hspace=0.15, wspace=0.5)
        gs0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, width_ratios=[1])
        gs00 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2,
                                                subplot_spec=gs0[0])

        ax1 = fig.add_subplot(gs00[0, 0])
        ax2 = fig.add_subplot(gs00[0, 1])

        for i, nb in enumerate(nominal_biases[:]):
            c_ds_name = 'sweep_{0}'.format(i)
            f_ds_name = 'fit_cdv_{0}'.format(i)
            print('Plotting for nominal bias {0} V'.format(nb))
            pband_color = lighten_color(nb_colors[i], 0.25)

            dlcp_ds = hf['dlcp']
            data = np.array(dlcp_ds[c_ds_name], dtype=dlcp_type)
            osc_level = data['osc_level']
            capacitance = data['C'] * 1E12

            fittings_ds = hf['fittings']
            data_fit = np.array(fittings_ds[f_ds_name], dtype=cfit_type)
            xpred = data_fit['osc_level']
            ypred = data_fit['C_fit']
            lpb = data_fit['lpb']
            upb = data_fit['upb']
            #            ax1.fill_between(xpred*1000,lpb,upb, color=pband_color)
            ax1.plot(osc_level * 1000, capacitance, 'o', color=nb_colors[i], fillstyle='none')
            ax1.plot(xpred * 1000, ypred, color=nb_colors[i])

        ndl_data = np.array(hf['NDL'], dtype=ndl_type)
        xvar = ndl_data['xvar']
        ndl = ndl_data['NDL']
        ndl_err = ndl_data['NDL_err']

        ax2.errorbar(xvar, ndl, yerr=ndl_err, ecolor='tab:blue',
                     capsize=2.0, capthick=1.0, fillstyle='none',
                     marker='o', mew=1.25)
        ax2.set_yscale('log')
        #    ax2.set_ylim(1E12,1E15)

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="7.5%", pad=0.03)
        cbar = fig.colorbar(scalar_maps, cax=cax)
        cbar.set_label(r'Nominal Bias (V)', rotation=90)

        ax1.set_xlabel('$\delta V$ (mV p-p)')
        ax1.set_ylabel('Capacitance (pF)')
        ax2.set_xlabel(r'$\epsilon A/C$ (cm)')
        ax2.set_ylabel(r'Charge Density (cm$^{-3}$)', color='tab:blue')

        xfmt = ScalarFormatter(useMathText=True)
        xfmt.set_powerlimits((-3, 3))
        ax2.xaxis.set_major_formatter(xfmt)

        ax2.yaxis.tick_right()
        ax2.yaxis.set_ticks_position('both')
        ax2.yaxis.set_label_position('right')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        ax1.yaxis.set_major_locator(mticker.MaxNLocator(5, prune=None))
        ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

        ax1.xaxis.set_major_locator(mticker.MaxNLocator(5, prune=None))
        ax1.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))

        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.show()

        filetag = os.path.splitext(os.path.basename(data_file))[0]

        fig.savefig(os.path.join(data_folder, '{}.png'.format(filetag)), dpi=600)
        fig.savefig(os.path.join(data_folder, '{}.eps'.format(filetag)), dpi=600,
                    format='eps')
