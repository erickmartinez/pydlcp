from datetime import datetime
from datetime import timedelta
from pydlcp import datastorage
import numpy as np
from typing import List


class BTS:
    """
    This class represents a bias-temperature stress (BTS) measurement

    Attributes
    ----------
    _accumulatedStressTime: float
        The current accumulated stress time in seconds
    _bias: float
        The stress bias in volts
    _intervalProgress: int
        A counter for the number of interval stresses being applied so far
    _intervals: int
        The number of intervals needed to complete a stress time of _maxTime
    _maxTime: float
        The total stress time in seconds
    _startTime: datetime
        The startTime of the measurement
    _start_bts_interval: datetime
        The time at which the current bias stress was initiated
    _status: str
        The status of the measuement:
        status must be a value in
        ['idle', 'heating_up', 'cooling_down', 'running_stress', 'running_cv', 'running_dlcp', 'finished']

    Methods
    -------
    set_start_time(self, start_datetime: datetime)
        Sets the start of the BTS measurement

    start_bts_interval(self):
        The time at which the last bias stress was initiated

    time_delta_total(self, current_datetime: datetime) -> float:
        Returns a timedelta in seconds between the provided datetime and the datetime at the beginning of the
        experiment.

    time_delta_bts(self, current_datetime: datetime) -> float:
        Returns a timedelta in seconds between the provided datetime and the datetime at the beginning of the last
        voltage stress.

    temperature(self) -> int:
        The stress temperature in °C

    bias(self) -> float:
        The stress bias in (V)

    stress_interval(self) -> int:
        The stress interval in seconds

    temperature_sampling_interval(self) -> int:
        The temperature sampling interval in seconds

    accumulate_interval(self, dt: float):
        Increases the interval counter and adds the estimated stress time to the accumulatedStresTime property.

    max_time(self):
        The target total stress time in seconds

    append_device(self, device: str, pin: int, h5_storage: datastorage.H5Store, clean: bool = True):
        Appends a device to the BTS measurement. A devices is a physical MIS structure. A convened named is used and
        a respective pin in the measurement unit must be provided, as well as the H5 datastorage for this device.

    get_device_storage(self, device: str, clean: bool = True) -> datastorage.H5Store:
        Returns the datastorage.H5Storage for the selected device.
    """
    _cleanDevices = []
    _cleanPins = []
    _cleanH5Storage = []
    _contaminatedDevices = []
    _contaminatedPins = []
    _contaminatedH5Storage = []
    _start_bts_interval: datetime = None
    _startTime: datetime = None
    _validStatus = ['idle', 'heating_up', 'cooling_down', 'running_stress', 'running_cv', 'running_dlcp', 'finished']

    def __init__(self, temperature: float, bias: float, stress_interval: int, temperature_sampling_interval: int,
                 max_time: float):
        """
        Parameters
        ----------
        temperature: float
            The temperature stress in °C
        bias: float
            The bias stress in volts
        stress_interval: int
            The stress interval in seconds. The sample will be subject to x number of stress intervals in order to
            complete a total stress time defined by 'max_time
        temperature_sampling_interval': int
            The interval at which the temperature (and current) will be sampled, given in seconds.
        max_time: float
            The total stress time in hours.
        """
        self._temperature: float = temperature
        self._bias: float = bias
        self._status: str = 'idle'
        self._accumulatedStressTime: float = 0
        self._stressInterval: int = stress_interval
        self._temperatureSamplingInterval: int = temperature_sampling_interval
        self._maxTime: float = max_time * 3600
        self._intervals = int(np.ceil(self._maxTime / stress_interval))
        self._intervalProgress = 0

    def set_start_time(self, start_datetime: datetime):
        """
        Sets the start time for the BTS Measurement

        Parameters
        ----------
        start_datetime: datetime
            The datetime object representing the date at which the BTS measurement started

        Raises
        -------
        TypeError
            If The argument is not an instance of datetime
        """
        if not isinstance(start_datetime, datetime):
            raise TypeError('The argument should be an instance of \'datetime\'.')
        self._startTime = start_datetime

    @property
    def start_bts_interval(self):
        return self._start_bts_interval

    @start_bts_interval.setter
    def start_bts_interval(self, start_datetime: datetime):
        """
        Sets the datetime object representing the time at which the last bias stress was initiated.

        Parameters
        ----------
        start_datetime: datetime
            A datetime object representing the date and time at which the last bias stress started.

        Raises
        -------
        TypeError
            If the start_datetime is not an instance of datetime.
        """
        if not isinstance(start_datetime, datetime):
            raise TypeError('The argument should be an instance of \'datetime\'.')
        self._start_bts_interval = start_datetime

    def time_delta_total(self, current_datetime: datetime) -> float:
        """
        Returns the timedelta between the provided datetime and the datetime at the beginning of the measurement

        Parameters
        ----------
        current_datetime: datetime
            The date and time

        Returns
        -------
        float
            The difference in time (in seconds).

        Raises
        ------
        TypeError
            If the provided current_datetime is not an instance of datetime.
        """
        if not isinstance(current_datetime, datetime):
            raise TypeError('The argument should be an instance of \'datetime\'.')
        dt: timedelta = current_datetime - self._startTime
        return dt.total_seconds()

    def time_delta_bts(self, current_datetime: datetime) -> float:
        """
        Returns the timedelta between the provided datetime and the datetime at the beginning of the last
        bias stress.

        Parameters
        ----------
        current_datetime: datetime
            The date and time

        Returns
        -------
        float
            The difference in time (in seconds).

        Raises
        ------
        TypeError
            If the provided current_datetime is not an instance of datetime.
        """
        if not isinstance(current_datetime, datetime):
            raise TypeError('The argument should be an instance of \'datetime\'.')
        dt: timedelta = current_datetime - self._start_bts_interval
        return dt.total_seconds()

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def bias(self) -> float:
        return self._bias

    @property
    def stress_interval(self) -> int:
        return self._stressInterval

    @property
    def temperature_sampling_interval(self) -> int:
        return self._temperatureSamplingInterval

    @property
    def status(self) -> str:
        if self._intervalProgress >= self._intervals:
            return 'finished'
        return self._status

    def accumulate_interval(self, dt: float):
        """
        After bias-temperature stress interval has completed accumulates the total stress time and increases the
        interval counter

        Parameters
        ----------
        dt: float
            The amount of time kept under stress during the present interval (seconds).

        Raises
        ------
        TypeError
            If dt is not float or int
        ValueError
            If dt < 0
        """
        if not isinstance(dt, float) and not isinstance(dt, int):
            raise TypeError('The time interval should be either int or float. Provided: {0}.'.format(dt))

        if dt < 0:
            raise ValueError('The time interval should be a positive real. Provided: {0}.'.format(dt))
        self._intervalProgress += 1
        self._accumulatedStressTime += dt

    @status.setter
    def status(self, status: str):
        if status not in self._validStatus:
            raise ValueError('Status \'{0}\' is not valid.'.format(status))
        self._status = status

    @property
    def max_time(self):
        return self._maxTime

    @property
    def interval_progress(self) -> int:
        return self._intervalProgress

    def append_device(self, device: str, pin: int, h5_storage: datastorage.H5Store, clean: bool = True):
        """
        Adds a device to the BTS experiment

        Parameters
        ----------
        device: str
            The device id
        pin: int
            The pin in the test unit, to which the device is attached.
        h5_storage: datastorage.H5Store
            The h5Store on which the data is saved.
        clean: bool
            True if the device does not contain sodium, false otherwise.
        """
        if clean:
            self._cleanPins.append(pin)
            self._cleanDevices.append(device)
            self._cleanH5Storage.append(h5_storage)
        else:
            self._contaminatedPins.append(pin)
            self._contaminatedDevices.append(device)
            self._contaminatedH5Storage.append(h5_storage)

    @property
    def clean_pins(self) -> List[int]:
        return self._cleanPins

    @property
    def contaminated_pins(self) -> List[int]:
        return self._contaminatedPins

    @property
    def clean_devices(self) -> List[str]:
        return self._cleanDevices

    @property
    def contaminated_devices(self) -> List[str]:
        return self._contaminatedDevices

    @property
    def accumulated_stress_time(self) -> float:
        return self._accumulatedStressTime

    def get_device_storage(self, device: str, clean: bool = True) -> datastorage.H5Store:
        """
        Returns the H5 Datastore to the selected device

        Parameters
        ----------
        device: str
            The device id
        clean: bool
            True if the device does not contain sodium contamination, false otherwise.

        Returns
        -------
        datastorage.H5Store
            The H5 Datastore to the selcted device
        """
        if clean:
            return self._cleanH5Storage[device == self._cleanDevices]
        else:
            return self._contaminatedH5Storage[device == self._contaminatedDevices]
