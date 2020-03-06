import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import svd
from scipy.linalg import norm
from scipy.optimize import OptimizeResult
import h5py
from datetime import datetime
import pydlcp.confidence as cf

dlcp_type = np.dtype([('osc_level', 'd'), ('bias', 'd'), ('nominal_bias', 'd'), ('V', 'd'), ('C', 'd')])
cfit_type = np.dtype([('osc_level', 'd'), ('C_fit', 'd'), ('lpb', 'd'), ('upb', 'd')])
ndl_type = np.dtype([('xvar', 'd'), ('xvar_err', 'd'), ('NDL', 'd'), ('NDL_err', 'd')])


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
            self._nominal_biases = np.arange(start=nb_start, stop=nb_stop + nb_step, step=nb_step)

    def get_dlcp_sweep_at_index(self, index: int) -> np.ndarray:
        if index < 0 or index > len(self._nominal_biases):
            raise ValueError('The index is out of bounds for the nominal bias.')
        ds_name = 'sweep_{0}'.format(index)
        with h5py.File(self._h5File, 'r') as hf:
            dlcp_ds = hf['dlcp']
            data = np.array(dlcp_ds[ds_name], dtype=dlcp_type)
        return data

    def estimate_nl(self) -> np.ndarray:
        ndl_data = np.empty(len(self._nominal_biases), dtype=ndl_type)
        for i, nb in enumerate(self._nominal_biases):
            _, fit_results = self.fit_dlcp_at_index(i)
            print('x_var = {0:.2E} cm, NDL = {1:.2E} cm^-3.'.format(fit_results['xvar'], fit_results['NDL']))
            ndl_data[i] = (fit_results['xvar'], fit_results['xvar_err'], fit_results['NDL'], fit_results['NDL_err'])
        ds_name = 'NDL'
        # If it does not exist, create it
        iso_date = datetime.now().astimezone().isoformat()
        with h5py.File(self._h5File, 'a') as hf:
            if ds_name not in hf:
                group_ds = hf.create_dataset(name=ds_name, shape=ndl_data.shape, compression='gzip', dtype=ndl_type)
            else:
                del hf[ds_name]
                group_ds = hf.create_dataset(name=ds_name, shape=ndl_data.shape, compression='gzip', dtype=ndl_type)
            group_ds[...] = ndl_data
            group_ds.attrs['iso_date'] = iso_date
        return ndl_data

    def fit_dlcp_at_index(self, index: int):
        """

        Parameters
        ----------
        index: int
            The index at which the data is fitted
        """
        area_cm = self._electrode_area_mm * 1E-2
        area_cm_err = self._electrode_area_error_mm * 1E-2

        all_tol = np.finfo(np.float64).eps
        # Get the data from the h5 file
        data = np.array(self.get_dlcp_sweep_at_index(index), dtype=dlcp_type)
        # Extract osc_level and capacitance
        osc_level = data['osc_level']
        capacitance = data['C']*1E12
        # Fit the curve
        res: OptimizeResult = least_squares(self.fobj,
                                            np.array([np.mean(capacitance), np.mean(np.gradient(capacitance))]),
                                            jac=self.fun_jac,
                                            args=(osc_level, capacitance),
                                            # bounds=([1E-50, -np.inf], [np.inf, 0.0]),
                                            xtol=all_tol,
                                            ftol=all_tol,
                                            gtol=all_tol,
                                            verbose=0)
        # Get the optimized parameters
        popt = res.x
        # Estimate the covariance matrix
        ysize: int = len(res.fun)
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

        # Estimate the 95% confidence interval
        ci = cf.confint(ysize, popt, pcov)
        # Estimate the prediction (potentially using more points)
        xpred = np.linspace(np.amin(osc_level), np.amax(osc_level), 100)
        ypred, lpb, upb = cf.predint(xpred, osc_level, capacitance, self.model, res, mode='observation')

        # Estimate the drive level
        ndl = self.n_dl(c0=popt[0], c1=popt[1], er=7.0, area_cm=area_cm)
        # Estimate the error to the drive level
        v0 = np.array([3 * pcov[0][0] / popt[0],
                       -pcov[1][0] / popt[1],
                       np.sqrt(np.abs(3 * pcov[1][1] / popt[0] / popt[1])),
                       area_cm_err / area_cm])
        ndl_err = ndl * norm(v0)
        # Estimate the depth
        xvar = self.xvariation(c0=popt[0], er=7.0, area_cm=area_cm)

        # Store the fit in the h5 file
        metadata = {
            'xvar': xvar,
            'xvar_err': 0.0,
            'NDL': ndl,
            'NDL_err': ndl_err,
            'popt': popt,
            'pcov': pcov,
            'ci': ci,
            'electrode_area_mm': self._electrode_area_mm,
            'electrode_area_error_mm': self._electrode_area_error_mm,
        }

        fit_data = np.empty(len(xpred), dtype=cfit_type)
        for i, x, y, yl, yu in zip(range(len(xpred)), xpred, ypred, lpb, upb):
            fit_data[i] = (x, y, yl, yu)
        self._append_data_set(data=fit_data, index=index, metadata=metadata)
        return fit_data, metadata

    def _append_data_set(self, data: np.ndarray, index: int = 0, metadata=None):
        """
        Appends a dataset to the selected group

        Parameters
        ----------
        data: np.ndarray
            The data to store in the dataset
        index: int
            The index for the dataset

        Raises
        ------
        ValueError:
            If the dataset already existed in the group
        """

        group_name = 'fittings'
        dataset_name = 'fit_cdv'

        if metadata is None:
            metadata = {}
        with h5py.File(self._h5File, 'a') as hf:
            if group_name not in hf:
                hf.create_group(group_name)

        # Get the current timestamp
        iso_date = datetime.now().astimezone().isoformat()
        ds_name = '{0}_{1:d}'.format(dataset_name, index)

        # append to the dataset
        with h5py.File(self._h5File, 'a') as hf:
            group = hf.get(group_name)
            if ds_name not in group:
                group_ds = group.create_dataset(name=ds_name, shape=data.shape, compression='gzip', dtype=cfit_type)
            else:
                # print('The dataset \'{0}\' already existed in \'{1}\'. Deleting...'.format(ds_name, group_name))
                del hf[group_name][ds_name]
                group_ds = group.create_dataset(name=ds_name, shape=data.shape, compression='gzip', dtype=cfit_type)
            group_ds[...] = data
            group_ds.attrs['iso_date'] = iso_date
            if metadata is not None:
                for key, val in metadata.items():
                    group_ds.attrs[key] = val

    @staticmethod
    def model(dv: [float, np.ndarray], b: np.ndarray):
        """
            Model for the DLCP capacitance
            Appl. Phys. Lett. 110, 203901 (2017
            
            C = C0 + C1 dV + 2 * (C1^2/C0) dV^2 + 5 * (C1^3 / C0^2) dV^3

            Parameters
            ----------
            b: np.ndarray
                The coefficients of the polynomial
            dv: np.ndarray
                The peak-to-peak oscillator level

            Returns
            -------
            np.ndarray
                The polynomial evaluated at the point x
            """
        return b[0] + b[1] * dv + 2 * (b[1] ** 2 / b[0]) * np.power(dv, 2) + 5 * (b[1] ** 3 / b[0] ** 2) * np.power(dv,
                                                                                                                    3)

    @staticmethod
    def fobj(b: np.ndarray, dv: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        A polynomial model for the capacitance

        Parameters
        ----------
        b: np.ndarray
            The coefficients of the polynomial
        dv: np.ndarray
            The x values to evaluate the polynomial
        c: np.ndarray
            The experimental values


        Returns
        -------
        np.ndarray
            The residual for polynomial evaluated at the point x
        """
        return b[0] + b[1]*dv + 2*((b[1]**2)/b[0])*np.power(dv, 2) + 5*(b[1]**3 / b[0] ** 2) * np.power(dv, 3.0) - c

    @staticmethod
    def fun_jac(b: np.ndarray, dv: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Estimates the jacobian of the objective function

        Parameters
        ----------
        b: np.ndarray
            The coefficients of the model
        dv: np.ndarray
            The oscillator levels at which to evaluate the jacobian
        y: np.ndarray
            The experimental capacitance (not used).

        Returns
        -------
        np.ndarray
            The jacobian matrix

        """
        jac = np.empty((len(dv), 2))
        c = b[1] / b[0]
        for i, v in enumerate(dv):
            jac[i] = (1.0 - 2.0 * (c * v) ** 2.0 - 10.0 * (c * v) ** 3.0,
                      v + 4.0 * c * (v ** 2.0) + 15.0 * (c ** 2.0) * (v ** 3.0))
        return jac

    @staticmethod
    def xvariation(c0: float, er: float, area_cm: float):
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

        return abs(x)

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

        return abs(ndl)
