import h5py
import numpy as np
from datetime import datetime

vcr_type = np.dtype([('V', 'd'), ('C', 'd'), ('R', 'd')])
tti_type = np.dtype([('time', 'd'), ('temperature', 'd'), ('current', 'd')])
dlcp_type = np.dtype([('osc_level', 'd'), ('bias', 'd'), ('nominal_bias', 'd'), ('V', 'd'), ('C', 'd'), ('R', 'd')])


class DLCPDataStore:
    """
    This class define methods to store the results from a DLCP measurement in an h5 file
    """

    def __init__(self, file_path: str):
        """
        Parameters
        ----------
        file_path: str
            The name of the h5 file to store data to.
        """
        self._file_path = file_path
        self._create_h5()

    @property
    def file_path(self) -> str:
        return  self._file_path

    def _create_h5(self):
        """
        Creates the h5 file and appends the basic groups
        """
        with h5py.File(self._file_path, 'w') as hf:
            hf.create_group("cv")  # Store bias temperature stress capacitance data
            hf.create_group("dlcp")  # Store DLCP data

    def save_cv(self, cv_data: np.ndarray):
        """
        Saves a Capacitance-Voltage sweep in the h5 structure
        Parameters
        ----------
        cv_data: vcr_type
            The CV sweep data

        Raises
        ------
        ValueError:
            If the data set already existed in the group
        """
        self._append_data_set(group_name='cv', data=cv_data, dtype=vcr_type)

    def save_dlcp(self, dlcp_data: np.ndarray, nominal_bias: float):
        """
        Append the DLCP sweep to the h5 file

        Parameters
        ----------
        dlcp_data: dlcp_type
            The dlcp data from the scan
        nominal_bias: float
            The nominal bias for the measurement

        Raises
        ------
        ValueError:
            If the data set already existed in the group
        """
        metadata = {'nominal_bias': nominal_bias}
        self._append_data_set(group_name='dlcp', data=dlcp_data, dtype=dlcp_type, metadata=metadata)

    def _append_data_set(self, group_name: str, data: np.ndarray, dtype=None, metadata: dict = {}):
        """
        Appends a dataset to the selected group

        Parameters
        ----------
        group_name: str
            The name of the group to append the dataset to.
        data: np.ndarray
            The data to store in the dataset
        dtype: np.dtype
            The type of data to store

        Raises
        ------
        ValueError:
            If the dataset already existed in the group
        """
        # Get the current timestamp
        iso_date = datetime.now().astimezone().isoformat()
        # Check the number of datasets in the group
        n = self.count_data_sets(group_name)
        ds_name = 'sweep_{0:d}'.format(n)
        # append to the dataset
        with h5py.File(self._file_path, 'a') as hf:
            group = hf.get(group_name)
            if ds_name not in group:
                group_ds = group.create_dataset(name=ds_name, shape=data.shape, compression='gzip', dtype=dtype)
                group_ds[...] = data
                group_ds.attrs['iso_date'] = iso_date
                for key, val in metadata.items():
                    group_ds.attrs[key] = val
            else:
                raise ValueError('The dataset \'{0}\' already existed in \'{1}\''.format(ds_name, group_name))

    def _append_resizeble_dataset(self, group_name: str, data_set_name: str, data, dtype=None):
        """
        Appends data to a resizable dataset in the H5 File

        Parameters
        ----------
        group_name: str
            The name of the group to append the dataset to.
        data_set_name: str
            The name of the dataset to append.
        data: np.ndarray
            The data to store in the dataset
        dtype: np.dtype
            The type of data to store
        """
        if not isinstance(data, np.ndarray) and not isinstance(data, list):
            data = np.array([data])
        with h5py.File(self._file_path, 'a') as hf:
            group = hf.get(group_name)
            if data_set_name not in group:
                group_ds = group.create_dataset(name=data_set_name, shape=(0, data.shape[0]), compression='gzip',
                                                chunks=True, maxshape=(None, data.shape[0]), dtype=dtype)
            else:
                group_ds = group.get(data_set_name)

            n = group_ds.shape[0]
            group_ds.resize(n + 1, axis=0)
            group_ds[-data.shape[0]:] = data

    def metadata(self, metadata: dict, group="/"):
        """
        Saves a dictionary with the measurement metadata to the specified dataset/group.

        Parameters
        ----------
        metadata: dict
            A dictionary with the metadata to save
        group: str
            The dataset/group to save the attribures to.
        """
        if not isinstance(metadata, dict):
            raise TypeError('The argument must be of type ')
        with h5py.File(self._file_path, 'a') as hf:
            group = hf.get(group) if group != "/" else hf
            for key, val in metadata.items():
                group.attrs[key] = val

    def get_metadata(self, group="/") -> dict:
        """
        Returns the attributes of a selected group.

        Parameters
        ----------
        group: str
            The group to get the attributes from

        Returns
        -------
        dict:
            A dictionary with the attributes of the dataset/group
        """
        with h5py.File(self._file_path, 'r') as hf:
            metadata = dict(hf.get(group).attrs)
        return metadata

    def create_resizeable_dataset(self, name: str, size: (int, int), group_name: str, dtype=None):
        """
        Creates a resizable dataset in the group 'group_name'.

        Parameters
        ----------
        name: str
            The name of the dataset
        size: (int, int)
            The shape of the dataset
        group_name: str
            The name of the group to save the dataset to
        dtype: np.dtype
            The type of data to be stored
        """
        if not isinstance(name, str):
            raise TypeError('Name should be an instance of str')
        with h5py.File(self._file_path, 'a') as hf:
            if group_name not in hf:
                hf.create_group(group_name)
            group = hf.get(group_name)
            group.create_dataset(name=name, shape=size, dtype=dtype, compression='gzip', chunks=True, maxshape=(None,))

    def count_data_sets(self, path: str):
        """
        Finds the number of datasets in the selected group

        Parameters
        ----------
        path: str
            The relative path to the dataset in the h5 file

        Returns
        -------
        int
            The number of datasets at that relative path
        """
        with h5py.File(self._file_path, 'a') as hf:
            try:
                n = len(list(hf[path]))
            except Exception as e:
                return 0
            else:
                return n

    def create_fixed_data_set(self, name: str, size: (int, int), group_name: str, dtype=None):
        """
        Creates a non-resizable dataset in the group 'group_name'.

        Parameters
        ----------
        name: str
            The name of the dataset
        size: (int, int)
            The shape of the dataset
        group_name: str
            The name of the group to save the dataset to
        dtype: np.dtype
            The type of the data being store
        """
        if not isinstance(name, str):
            raise TypeError('Name should be an instance of str')
        with h5py.File(self._file_path, 'a') as hf:
            if group_name not in hf:
                hf.create_group(group_name)
            group = hf.get(group_name)
            group.create_dataset(name=name, shape=size, dtype=dtype, compression='gzip')
