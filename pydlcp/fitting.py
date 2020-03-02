import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import svd
from scipy.linalg import norm
from typing import List
import h5py
import datetime
import pydlcp.confidence as cf
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.pyplot as plt


dlcp_type = np.dtype([('osc_level', 'd'), ('bias', 'd'), ('nominal_bias', 'd'), ('V', 'd'), ('C', 'd')])


class Fitting:

    _nominal_biases: np.ndarray = None

    def __init__(self, h5_file: str, electrode_area_mm: float, electrode_area_error_mm: float = 0.0):
        self._h5File: str = h5_file
        self._electrode_area_mm: float = electrode_area_mm
        self._electrode_area_error_mm: float = electrode_area_error_mm
        with h5py.File(h5_file) as hf:
            dlcp_ds = hf['dlcp']
            nb_start = float(dlcp_ds.attrs['nominal_bias_start'])
            nb_step = float(dlcp_ds.attrs['nominal_bias_step'])
            nb_stop = float(dlcp_ds.attrs['nominal_bias_stop'])
            self._nominal_biases = np.arange(start=nb_start, stop=nb_stop+nb_step, step=nb_step)

    def get_dlcp_sweep_at_index(self, index) -> np.ndarray:
        if index < 0 or index > len(self._nominal_biases):
            raise ValueError('The index is out of bounds for the nominal bias.')
        ds_name = 'sweep_{0}'.format(index)
        with h5py.File(self._h5File, 'r') as hf:
            dlcp_ds = hf['dlcp']
            data = np.array(dlcp_ds[ds_name], dtype=dlcp_type)
        return data

    def fit_dlcp_at_index(self, index: int):
        """

        Parameters
        ----------
        index: int
            The index at which the data is fitted
        """
        area_cm = self._electrode_area_mm * 1E-2
        area_cm_err = self._electrode_area_error_mm * 1E-2

        color_palette = 'winter'

        data = self.get_dlcp_sweep_at_index(index)

        nominal_biases = data['nominal_bias']

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

        nfiles = data['nominal_bias'].size()

        ndl = np.empty(nfiles)
        ndl_err = np.empty(nfiles)
        xvar = np.empty(nfiles)
        all_tol = np.finfo(np.float64).eps

        osc_level = data['osc_level']
        capacitance = data['capacitance']

        res = least_squares(self.fobj, np.array([1.0, 1.0, 0.0, 0.0]), jac=self.fun_jac,
                            args=(osc_level, capacitance),
                            xtol=all_tol,
                            ftol=all_tol,
                            gtol=all_tol,
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
        pcov = np.dot(VT.T / s ** 2, VT)
        pcov = pcov * s_sq

        if pcov is None:
            # indeterminate covariance
            print('Failed estimating pcov')
            pcov = np.zeros((len(popt), len(popt)), dtype=float)
            pcov.fill(np.inf)

        xpred = np.linspace(np.amin(osc_level), np.amax(osc_level), 100)

        ci = cf.confint(ysize, popt, pcov)
        ypred, lpb, upb = cf.predint(xpred, osc_level, capacitance, self.model,
                                     res, mode='observation')

        i = index

        ax1.plot(xpred * 1000, ypred, color=nb_colors[i])
        pband_color = [(c * 2) % 1 for c in nb_colors[i]]
        #        ax1.fill_between(lpb,upb, alpha=.5, color=pband_color)

        ndl[i] = self.n_dl(c0=popt[0], c1=popt[1], er=7.0, area_cm=area_cm)

        v0 = np.array([3 * pcov[0][0] / popt[0],
                       -pcov[1][0] / popt[1],
                       np.sqrt(np.abs(3 * pcov[1][1] / popt[0] / popt[1])),
                       area_cm_err / area_cm])

        ndl_err[i] = ndl[i] * norm(v0)

        xvar[i] = self.xvariation(c0=popt[0], er=7.0, area_cm=area_cm)

    def _append_data_set(self, data: np.ndarray, dtype=None, metadata: dict = {}):
        """
        Appends a dataset to the selected group

        Parameters
        ----------
        data: np.ndarray
            The data to store in the dataset
        dtype: np.dtype
            The type of data to store

        Raises
        ------
        ValueError:
            If the dataset already existed in the group
        """
        # First check if we have a '/fittings' group in the h5 file
        # If it does not exist, create it
        group_name = 'fittings'
        with h5py.File(self._file_path, 'a') as hf:
            if group_name not in hf:
                hf.create_group(group_name)
            group = hf.get(group_name)
            group.create_dataset(name=data['name'], shape=data['size'], dtype=dtype, compression='gzip')

        # Get the current timestamp
        iso_date = datetime.now().astimezone().isoformat()
        # Check the number of datasets in the group
        n = self.count_data_sets(group_name)
        ds_name = 'fitting_{0:d}'.format(n)

        # append to the dataset
        with h5py.File(self._h5File, 'a') as hf:
            group = hf.get(group_name)
            if ds_name not in group:
                group_ds = group.create_dataset(name=ds_name, shape=data.shape, compression='gzip', dtype=dtype)
                group_ds[...] = data
                group_ds.attrs['iso_date'] = iso_date
                for key, val in metadata.items():
                    group_ds.attrs[key] = val
            else:
                raise ValueError('The dataset \'{0}\' already existed in \'{1}\''.format(ds_name, group_name))


    def model(self, x: [float,np.ndarray], b: np.ndarray):
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
        return b[0] + b[1] * x + b[2] * np.power(x, 2) + b[3] * np.power(x, 3)


    def fobj(self, b: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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
        return b[0] + b[1] * x + b[2] * np.power(x, 2) + b[3] * np.power(x, 3) - y

    def fun_jac(self, b: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        jac = np.empty((len(x), 4))
        for i, xi in enumerate(x):
            jac[i] = (1.0, xi, xi ** 2, xi ** 3)
        return jac

    def xvariation(self, c0: float, er: float, area_cm: float):
        """
        Estimates the quantity

        eps*A / c0 = x_e + eps*F_e/rho_e

        which corresponds to variations in the depletion width over approximately
        the same distance scale.

        Parameters
        ----------
        c0: float
            The fitted value of c0 in pico Farads (C^2/J)
        er: float
            The relative permittivity of the dielectric
        area_cm: float
            The area of the device in cm^2

        Returns
        -------
        float:
            eps*A/c0 in cm
        """
        # e0 = 8.854187817620389e-14 C^2 / J / cm
        # q = 1.6021766208e-19
        x = er * 8.854187817620389 * area_cm / c0 / 100

        return x

    @staticmethod
    def n_dl(c0: float, c1: float, er: float, area_cm: float):
        """
        Estimates the drive level from the fitted values C0 and C1

        C = c0 + c1 dV + C2 (dV)^2 + ...

        Parameters
        ----------
        c0: float
            The fitted c0 in pico Farads (10^-12 C^2/J)
        c1: float
            The fitted c1 in pico Farads/V (10^-12 C^3/J^2)
        er: float
            The relative permittivity of the dielectric
        area_cm: float
            The area of the device in cm^2

        Returns
        -------
        float
            ndl
        """
        # e0 = 8.854187817620389e-14 C^2 / J / cm
        # q = 1.6021766208e-19

        qe = er * 8.854187817620389 * 1.6021766208  # x 1E-33
        ndl = -1.0E9 * np.power(c0, 3.0) / (2.0 * qe * area_cm * c1)

        return ndl
