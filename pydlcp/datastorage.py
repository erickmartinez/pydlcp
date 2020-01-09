
import h5py
import numpy as np

vcr_type = np.dtype([('V', 'd'), ('C', 'd'), ('R', 'd')])
tti_type = np.dtype([('time', 'd'), ('temperature', 'd'), ('current', 'd')])
dlcp_type = np.dtype([('osc_level', 'd'),
                      ('bias', 'd'),
                      ('nominal_bias', 'd'),
                      ('V', 'd'),
                      ('C', 'd'),
                      ('R', 'd')])


class H5Store:
    """
    This class provides methods to save and access BTS experimental results to an hdf5 file

    Attributes
    ----------
    _filename: str
        The path to the h5 file

    Methods
    -------
    _create_h5(self):
        Creates the h5 file and adds the basic structure containing 3 groups
        '/logs'
            A group to store all logs
        '/logs/temperature_and_current'
            A dataset that stores the temperautre and leakage current every x-seconds
        '/bts'
            A group to store al the bias-temperature CV results
        '/bts/time'
            A 1D dataset containing the timedelta at which each CV was taken
        '/bts/voltage'
            A Matrix containing the voltage sweep at each timedelta
        '/bts/capacitance'
            A Matrix containing the capacitance from the CV sweep
        '/bts/resistance'
            A Matrix containing Rs from the CV sweep.
        '/dlcp'
            A group containing al DLCP results
        '/dlcp/time'
            A 1D dataset containing the timedelta at which each DLCP sweep was taken
        '/dlcp/osc_level'
            A Matrix containing the osc_level sweep at each timedelta
        '/dlcp/bias'
            A Matrix containing the voltage sweep at each timedelta
        '/dlcp/nominal_bias'
            A Matrix containing the voltage sweep at each timedelta
        '/dlcp/voltage'
            A Matrix containing the voltage sweep at each timedelta
        '/dlcp/capacitance'
            A Matrix containing the capacitance from the DLCP sweep
        '/dlcp/resistance'
            A Matrix containing Rs from the CV sweep.

    create_temperature_log(self):
        Creates the resizable dataset "/logs/temperature_and_current" on the h5 file.

    log_temperature_current(self, time: float, temperature: float, current: float):
        Appends the temperature and current readings at time 't' to the '/logs/temperature_and_current' dataset

    get_bts_cv(self):
        Returns the stress_time, voltage, capacitance and resistance datasets in the /bts group

    get_dlcp_data(self):
        Returns the stress_time, osc_level, bias, nominal_bias, voltage, capacitance and resistance datasets in
        the /dlcp group

    get_bts_cv_at_time(self, time_idx: int) -> vcr_type:
        Returns a single CV sweep result at an specific time index

    get_bts_temperature(self):
        Returns the temperature log

    append_cv(self, time: float, cv_data: vcr_type):
        Appends a CV sweep result (vcr_type) to the /bts group

    append_dlcp(self, time: float, dlcp_data: dlcp_type):
        Appends a DLCP sweep result (dlcp_type) to the /dlcp group

    _append_resizeble_dataset(self, group_name: str, data_set_name: str, data):
        Appends data to a resizable dataset in the h5 file

    metadata(self, metadata: dict, group="/"):
        Appends dictionary key values as attributes to the selected group/dataset

    create_resizeable_dataset(self, name: str, size: (int, int), group_name: str, dtype=None):
        Creates a resizable dataset on a specific group in the h5 file

    create_fixed_dataset(self, name: str, size: (int, int), group_name: str, dtype=None):
        Creates a dataset that cannot be resized on a specific group in the h5 file.

    """

    def __init__(self, filename: str):
        """
        Parameters
        ----------
        filename: str
            The name of the h5 file to store data to.
        """
        self._filename = filename
        self._create_h5()

    def _create_h5(self):
        """
        Creates the h5 file and appends the basic groups
        """
        hf = h5py.File(self._filename, 'w')
        hf.create_group("logs")  # Store experimental logs
        hf.create_group("bts")  # Store bias temperature stress capacitance data
        hf.create_group("dlcp")  # Store DLCP data
        hf.close()

    def create_temperature_log(self):
        """
        Creates the 'temperature_and_current' dataset on the '/logs' dataset of the h5 file.
        """
        self.create_resizeable_dataset(name='temperature_and_current', size=(0,), group_name="logs", dtype=tti_type)

    def log_temperature_current(self, time: [float, int], temperature: [float, int], current: float):
        """
        Appends the tuple (time, temperature, current) to the 'temperature_and_current' dataset.

        Parameters
        ----------
        time: [float, int]
            The timedelta since the beginning of the log
        temperature: [float, int]
            The sample temperature in °C
        current: float
            The leakage current through the device.

        """
        data = np.array([(time, temperature, current)], dtype=tti_type)
        self._append_resizeble_dataset(group_name="/logs", data_set_name="temperature_and_current", data=data,
                                       dtype=tti_type)

    def get_bts_cv(self) -> dict:
        """
        Gets all the CV sweeps stored in the H5 file wrapped in a dictionary.

        Returns
        -------
        dict
            A dictionary containing the vector for 'stress_time', and the matrices 'voltage', 'capacitance' and
            'resistance'
        """
        with h5py.File(self._filename, 'r') as hf:
            if '/bts/time' in hf:
                stress_time = np.array(hf.get('/bts/time'))
                voltage = np.array(hf.get('/bts/voltage'))
                capacitance = np.array(hf.get('/bts/capacitance'))
                resistance = np.array(hf.get('/bts/resistance'))
                data = {
                    'stress_time': stress_time,
                    'voltage': voltage,
                    'capacitance': capacitance,
                    'resistance': resistance
                }
            else:
                data = None
        return data

    def get_dlcp_data(self) -> dict:
        """
        Gets al DLCP sweep data as dictionary.

        Returns
        -------
        dict:
            A dictionary containing the vector for 'stress_time', and the matrices 'osc_level', 'bias', 'nominal_bias',
            'voltage', 'capacitance' and 'resistance'
        """
        with h5py.File(self._filename, 'r') as hf:
            if '/dlcp/time' in hf:
                stress_time = np.array(hf.get('/dlcp/time'))
                osc_level = np.array(hf.get('/dlcp/osc_level'))
                bias = np.array(hf.get('/dlcp/bias'))
                nominal_bias = np.array(hf.get('/dlcp/nominal_bias'))
                voltage = np.array(hf.get('/dlcp/voltage'))
                capacitance = np.array(hf.get('/dlcp/capacitance'))
                resistance = np.array(hf.get('/dlcp/resistance'))
                data = {
                    'stress_time': stress_time,
                    'osc_level': osc_level,
                    'bias': bias,
                    'nominal_bias': nominal_bias,
                    'voltage': voltage,
                    'capacitance': capacitance,
                    'resistance': resistance
                }
            else:
                data = None
        return data

    def get_bts_cv_at_time(self, time_idx: int):
        """
        Returns a single CV sweep at a specific time index.

        Parameters
        ----------
        time_idx: int
            The row index corresponding to a specific time of the measurement.

        Returns
        -------
        np.ndarray(vcr_type)
            A numpy array of vcr_type
        """
        data = self.get_bts_cv()
        if data is None:
            return np.array([], dtype=vcr_type)
        if abs(time_idx) < data['stress_time'].shape[0]:
            n = len(data['voltage'])
            vcr_data: vcr_type = np.empty(n, dtype=vcr_type)
            for i, v, c, r in zip(range(n), data['voltage'][time_idx],
                                  data['capacitance'][time_idx],
                                  data['resistance'][time_idx]):
                vcr_data[i] = (v, c, r)
            return vcr_data
        else:
            return np.array([], dtype=vcr_type)

    def get_bts_temperature(self):
        """
        Returns the whole temperature log

        Returns
        -------
        np.ndarray(tti_type)
            The numpy array of type tti_type containing all temperature and leakage current logs.
        """
        with h5py.File(self._filename, 'r') as hf:
            if '/logs/temperature_and_current' in hf:
                data: tti_type = np.array(hf['/logs/temperature_and_current'], dtype=tti_type)
            else:
                data: tti_type = None
        return data

    def append_cv(self, time: [float, int], cv_data: vcr_type):
        """
        Appends a CV sweep to the H5 file

        Parameters
        ----------
        time: float
            The time delta at which the measurement was taken.
        cv_data: [vcr_type, np.ndarray(vcr_type)
            The CV sweep
        """
        self._append_resizeble_dataset(group_name="/bts", data_set_name="time", data=np.array([time]))
        self._append_resizeble_dataset(group_name="/bts", data_set_name="voltage", data=cv_data['V'])
        self._append_resizeble_dataset(group_name="/bts", data_set_name="capacitance", data=cv_data['C'])
        self._append_resizeble_dataset(group_name="/bts", data_set_name="resistance", data=cv_data['R'])

    def append_dlcp(self, time: [float, int], dlcp_data: dlcp_type):
        """
        Appends a DLCP sweep to the H5 file

        Parameters
        ----------
        time: [float, int]
            The time delta at which the DLCP measurement was performed
        dlcp_data: [dlcp_dtype, np.ndarray(dlcp_type)
            The DLCP data
        """
        self._append_resizeble_dataset(group_name="/dlcp", data_set_name="time", data=np.array([time]))
        self._append_resizeble_dataset(group_name="/dlcp", data_set_name="osc_level", data=dlcp_data['osc_level'])
        self._append_resizeble_dataset(group_name="/dlcp", data_set_name="bias", data=dlcp_data['bias'])
        self._append_resizeble_dataset(group_name="/dlcp", data_set_name="nominal_bias", data=dlcp_data['nominal_bias'])
        self._append_resizeble_dataset(group_name="/dlcp", data_set_name="voltage", data=dlcp_data['V'])
        self._append_resizeble_dataset(group_name="/dlcp", data_set_name="capacitance", data=dlcp_data['C'])
        self._append_resizeble_dataset(group_name="/dlcp", data_set_name="resistance", data=dlcp_data['R'])

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
        with h5py.File(self._filename, 'a') as hf:
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
        with h5py.File(self._filename, 'a') as hf:
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
        with h5py.File(self._filename, 'r') as hf:
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
        with h5py.File(self._filename, 'a') as hf:
            if group_name not in hf:
                hf.create_group(group_name)
            group = hf.get(group_name)
            group.create_dataset(name=name, shape=size, dtype=dtype, compression='gzip', chunks=True, maxshape=(None,))

    def create_fixed_dataset(self, name: str, size: (int, int), group_name: str, dtype=None):
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
        with h5py.File(self._filename, 'a') as hf:
            if group_name not in hf:
                hf.create_group(group_name)
            group = hf.get(group_name)
            group.create_dataset(name=name, shape=size, dtype=dtype, compression='gzip')
