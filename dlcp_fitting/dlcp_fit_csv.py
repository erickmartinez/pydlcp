# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:12:34 2020

@author: Erick
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import confidence as cf
from scipy.linalg import svd
import matplotlib as mpl
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from scipy.linalg import norm
import platform
import os

root_folder = r'G:\Shared drives\FenningLab2\Projects\PVRD1\ExpData\DLCP\SiNx\D233-p5'

area_mm = 1.3
area_cm = area_mm*1E-4
area_cm_err = 0.0

color_palette = 'winter'

ndl_type = np.dtype([('eA/C (cm)', 'd'), 
                     ('NDL (cm^-3)', 'd'),
                     ('NDL_err', 'd')])

def poly_model(x: [float,np.ndarray], b: np.ndarray):
    """
    A polynomial model for the capacitance
    
    Parameters
    ----------
    b: np.ndarray
        The coefficients of the polynomial
    x: np.ndarray
        The x values to evaluate the polynomial
    
    Returns
    -------
    np.ndarray
        The polynomial evaluated at the point x
    """
    return b[0] + b[1]*x + b[2]*np.power(x,2) + b[3]*np.power(x,3)





def n_dl(C0: float, C1: float, er: float, area_cm: float):
    """
    Estimates the drive level from the fitted values C0 and C1
    
    C = C0 + C1 dV + C2 (dV)^2 + ...
    
    Parameters
    ----------
    C0: float
        The fitted C0 in pico Farads (10^-12 C^2/J)
    C1: float
        The fitted C1 in pico Farads/V (10^-12 C^3/J^2)
    er: float
        The relative permittivity of the dielectric
    area_cm: float
        The area of the device in cm^2
    
    Returns
    -------
    float
        NDL
    """
    # e0 = 8.854187817620389e-14 C^2 / J / cm
    # q = 1.6021766208e-19
    
    qe = er*8.854187817620389*1.6021766208 # x 1E-33
    NDL = -1.0E9*np.power(C0,3.0)/(2*qe*area_cm*C1)
    
    return NDL


def xvariation(C0: float, er: float, area_cm: float):
    """
    Estimates the quantity
    
    eps*A / C0 = x_e + eps*F_e/rho_e
    
    which corresponds to variations in the depletion width over approximately 
    the same distance scale.
    
    Parameters
    ----------
    C0: float
        The fitted value of C0 in pico Farads (C^2/J)
    er: float
        The relative permittivity of the dielectric
    area_cm: float
        The area of the device in cm^2
    
    Returns
    -------
    float:
        eps*A/C0 in cm
    """
    # e0 = 8.854187817620389e-14 C^2 / J / cm
    # q = 1.6021766208e-19
    x = er*8.854187817620389*area_cm/C0/100
    
    return x
    
    

def files_with_extension(path: str,extension: str):
    """
    Gives a list of the files in the given directory that have the given extension
    
    Parameters
    ----------
    path: str
        The full path to the folder where the files are stored
    extension: str
        The extension of the files
    
    Returns
    -------
    List[str]
        A list containing the files
    """
    from os import listdir
    return [f for f in listdir(path) if f.endswith(extension)]

defaultPlotStyle = {'font.size': 14,
                     'font.family': 'Arial',
                     'font.weight': 'regular',
                    'legend.fontsize': 14,
                    'mathtext.fontset': 'custom',
                    'mathtext.rm': 'Times New Roman',
                    'mathtext.it': 'Times New Roman:italic',#'Arial:italic',
                    'mathtext.cal': 'Times New Roman:italic',#'Arial:italic',
                    'mathtext.bf': 'Times New Roman:bold',#'Arial:bold',
                    'xtick.direction' : 'in',
                    'ytick.direction' : 'in',
                    'xtick.major.size' : 4.5,
                    'xtick.major.width' : 1.75,
                    'ytick.major.size' : 4.5,
                    'ytick.major.width' : 1.75,
                    'xtick.minor.size' : 2.75,
                    'xtick.minor.width' : 1.0,
                    'ytick.minor.size' : 2.75,
                    'ytick.minor.width' : 1.0,
                    'ytick.right' : False,
                    'lines.linewidth'   : 2.5,
                    'lines.markersize'  : 10,
                    'lines.markeredgewidth'  : 0.85,
                    'axes.labelpad'  : 5.0,
                    'axes.labelsize' : 16,
                    'axes.labelweight' : 'regular',
                    'axes.linewidth': 1.25,
                    'axes.titlesize' : 16,
                    'axes.titleweight' : 'bold',
                    'axes.titlepad' : 6,
                    'figure.titleweight' : 'bold',
                    'figure.dpi': 100}


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



if __name__ == '__main__':
    if platform.system() == 'Windows':
        root_folder = u'\\\?\\' + root_folder
        
    mpl.rcParams.update(defaultPlotStyle)
    
    files = files_with_extension(path=root_folder, extension='csv')
    nfiles = len(files)
    sorted_files = []
    nominal_biases = np.empty(nfiles)
    
    # Order the file list by nominal bias

    for i,fn in enumerate(files):
        csv_file = os.path.join(root_folder,fn)
        print('Reading file: \'{0}\''.format(fn))
        df = pd.read_csv(filepath_or_buffer=csv_file, delimiter=',', 
                         index_col=0)
        
        try:
            nominal_biases[i] = df['nominal_bias'][0]
            sorted_files.append(dict(nominal_bias = df['nominal_bias'][0],
                                      filename = fn))
        except:
            print('nominal bias not found in \'{0}\'.'.format(fn))
            np.delete(nominal_biases,i)
        
    
    sorted_files = sorted(sorted_files, key = lambda i: i['nominal_bias']) 
    nominal_biases = np.sort(nominal_biases, axis=0)
    
    
    normalize = mpl.colors.Normalize(vmin=np.amin(nominal_biases), 
                                     vmax=np.amax(nominal_biases))
    
    cm = mpl.cm.get_cmap(color_palette)
    nb_colors = [cm(normalize(bb)) for bb in nominal_biases]
    scalar_maps = mpl.cm.ScalarMappable(cmap=cm, norm=normalize)
    
    fig = plt.figure()
    fig.set_size_inches(6.5,3.0,forward=True)
    fig.subplots_adjust(hspace=0.15, wspace=0.5)
    gs0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, width_ratios=[1])
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, 
                                            subplot_spec = gs0[0])

    ax1 = fig.add_subplot(gs00[0,0])
    ax2 = fig.add_subplot(gs00[0,1])
    
    ndl = np.empty(nfiles)
    ndl_err = np.empty(nfiles)
    xvar = np.empty(nfiles)
    
    all_tol = np.finfo(np.float64).eps
    
    ndl_data = np.empty(nfiles, dtype=ndl_type)

    
    for i,r in enumerate(sorted_files):
        fn = r['filename']
        csv_file = os.path.join(root_folder,fn)
        df = pd.read_csv(filepath_or_buffer=csv_file, delimiter=',', 
                         index_col=0)
        osc_level = np.array(df['osc_level'])
        capacitance = np.array(df['C']*1E12)
        
        idx_osc = osc_level >= 0.05
        osc_level = osc_level[idx_osc]
        capacitance = capacitance[idx_osc]
        
        ax1.plot(osc_level*1000,capacitance,'o',color=nb_colors[i], fillstyle='none')
        
        def fun(b: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
            """
            A polynomial model for the capacitance
            
            Parameters
            ----------
            b: np.ndarray
                The coefficients of the polynomial
            x: np.ndarray
                The x values to evaluate the polynomial
            y: np.ndarray
                The experimental values
            
            
            Returns
            -------
            np.ndarray
                The residual for polynomial evaluated at the point x
            """
            return b[0] + b[1]*x + b[2]*np.power(x,2) + b[3]*np.power(x,3) - y
        
        def fun_jac(b: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
            jac = np.empty((len(x), 4))
            for i, xi in enumerate(x):
                jac[i] = (1.0, xi, xi**2, xi**3)
            return jac
        
        res = least_squares(fun, np.array([1.0,1.0,0.0,0.0]), jac=fun_jac, 
                            args=(osc_level,capacitance),
                            xtol=all_tol,
                            ftol=all_tol,
                            gtol=all_tol,
#                            loss='soft_l1', f_scale=0.1,
                            max_nfev=nfiles*10000,
                            verbose=0)
        
        popt = res.x
        
        ysize = len(res.fun)
        cost = 2 * res.cost  # res.cost is half sum of squares!
        s_sq = cost / (ysize - popt.size)
        
        # Do Moore-Penrose inverse discarding zero singular values.
        _, s, VT = svd(res.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s**2, VT)
        pcov = pcov * s_sq
       

        if pcov is None:
            # indeterminate covariance
            print('Failed estimating pcov')
            pcov = np.zeros((len(popt), len(popt)), dtype=float)
            pcov.fill(np.inf)
        
        xpred = np.linspace(np.amin(osc_level), np.amax(osc_level), 100)
        
        ci = cf.confint(ysize,popt,pcov)
        ypred,lpb,upb = cf.predint(xpred, osc_level, capacitance, poly_model,
                               res, mode='observation')
        
        
        pband_color = lighten_color(nb_colors[i], 0.25)
        ax1.fill_between(xpred*1000,lpb,upb, color=pband_color)
        ax1.plot(osc_level*1000,capacitance,'o',color=nb_colors[i], fillstyle='none')
        ax1.plot(xpred*1000, ypred, color=nb_colors[i])
        
        ndl[i] = n_dl(C0=popt[0], C1=popt[1], er=7.0, area_cm=area_cm)
        
        v0 = np.array([3*pcov[0][0]/popt[0], 
                       -pcov[1][0]/popt[1], 
                       np.sqrt(np.abs(3*pcov[1][1]/popt[0]/popt[1])),
                       area_cm_err/area_cm])
    
        ndl_err[i] = ndl[i]*norm(v0)
        
        xvar[i] = xvariation(C0=popt[0], er=7.0, area_cm=area_cm)
        
        ndl_data[i] = (xvar[i], ndl[i], ndl_err[i])
        
    ax2.errorbar(xvar,ndl,yerr=ndl_err, ecolor='tab:blue',
                 capsize=2.0, capthick=1.0, fillstyle='none',
                 marker='o', mew=1.25)
    ax2.set_yscale('log')
#    ax2.set_ylim(1E12,1E15)
    
    divider = make_axes_locatable(ax1)
    cax     = divider.append_axes("right", size="7.5%", pad=0.03)
    cbar    = fig.colorbar(scalar_maps, cax=cax)
    cbar.set_label(r'Nominal Bias (V)', rotation=90)
    
    ax1.set_xlabel('Oscillator Amplitude (mV p-p)')
    ax1.set_ylabel('Capacitance (pF)')
    ax2.set_xlabel(r'$\epsilon A/C$ (cm)')
    ax2.set_ylabel(r'Charge Density (cm$^{-3}$)', color='tab:blue')
    
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((-3,3))
    ax2.xaxis.set_major_formatter(xfmt)
    
    ax2.yaxis.tick_right()
    ax2.yaxis.set_ticks_position('both')
    ax2.yaxis.set_label_position('right') 
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(5,prune=None))
    ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(5,prune=None))
    ax1.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    
     
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.show()
    
    fig.savefig(os.path.join(root_folder,'dlcp_test.png'), dpi=600)
    fig.savefig(os.path.join(root_folder,'dlcp_test.eps'), dpi=600, format='eps')
    
    df = pd.DataFrame(data=ndl_data)
    df.to_csv(path_or_buf=os.path.join(root_folder,'ndl_data.csv'), index=False)
        
        
        
    
