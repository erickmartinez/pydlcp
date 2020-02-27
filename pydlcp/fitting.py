import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import svd
from scipy.linalg import norm
from typing import List
import h5py
import datetime


dlcp_type = np.dtype([('osc_level', 'd'), ('bias', 'd'), ('nominal_bias', 'd'), ('V', 'd'), ('C', 'd')])


class Fitting:

    _nominal_biases: List[float] = []

    def __init__(self, h5_file: str, electrode_area_mm: float):
        self._h5File: h5py.File = h5_file
        self._electrode_area_mm: float = electrode_area_mm
        with h5py.File(h5_file) as hf:
            dlcp_ds = hf['dlcp']
            nb_start = float(dlcp_ds.attrs['nominal_bias_start'])
            nb_step = float(dlcp_ds.attrs['nominal_bias_step'])
            nb_stop = float(dlcp_ds.attrs['nominal_bias_stop'])
            self._nominal_biases = np.arange(start=nb_start, stop=nb_stop+nb_step, step=nb_step)

    def get_cv_sweep_at_index(self, index) -> np.ndarray:
        if index < 0 or index > len(self._nominal_biases):
            raise ValueError('The index is out of bounds for the nominal bias.')
        ds_name = 'sweep_{0}'.format(index)
        data = None
        with h5py.File(self._h5File, 'r') as hf:
            dlcp_ds = hf['dlcp']
            data = np.array(dlcp_ds[ds_name], dtype=dlcp_type)
        return data

    # def model(self, x: [float,np.ndarray], b: np.ndarray):
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
            group.create_dataset(name=name, shape=size, dtype=dtype, compression='gzip')

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
