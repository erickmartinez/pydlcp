"""
This class provides control functions for the DLCP system

@author Erick R Martinez Loran <erickrmartinez@gmail.com>
"""
import numpy as np
import configparser
import json
from datetime import datetime
from datetime import timedelta
import os
import logging
import pyvisa
from pydlcp import arduino_board, hotplate, errors, keithley, impedance_analyzer as ia, datastorage, bts
from apscheduler.schedulers.background import BackgroundScheduler
import platform
from typing import List

# Different data type definitions
ard_list = List[arduino_board.ArduinoBoard]
bts_list = List[bts.BTS]
hp_list = List[hotplate.Hotplate]
vcr_type = np.dtype([('V', 'd'), ('C', 'd'), ('R', 'd')])
dlcp_type = np.dtype([('osc_level', 'd'),
                      ('bias', 'd'),
                      ('nominal_bias', 'd'),
                      ('V', 'd'),
                      ('C', 'd'),
                      ('R', 'd')])


class Controller:
    """
    This class provides methods to control a BTS and DLCP experiment interacting with different instruments and saving
    to a h5 data store.

    Attributes
    ----------
    _activeCVUnit: List[str]
        The unit that the impedance analyzer is currently locked to.
    _arduinos:
        A list with handles to different ArduinoBoard instances controlling the different test units.
    _availableResources: List[str]
        A list of available pyvisa resources. Used to verify that the resource given by the constructor is available.
    _btsAcquisitions: bts_list
        A list with the different BTS instances representing a different experimental BTS condition.
    _cvSweepParams: dict
        A dictionary with the CV sweep acquisition parameters
    _data_paths: List[str]
        A list containing the paths where the output data a logs will be saved
    debug: bool
        True if we want to execute in debugging mode
    _configMeasurementRequiredOptions: dict
        A dictionary containing rules for validating the acquisition parameters and settings
    _configSystemRequiredOptions: dict
        A dictionary containing rules for validating the system configuration if it is not in its default state
    _dlcpParams: dict
        A dictionary with the DLCP acquisition parameters
    _finishedUnits: List[int]
        A list with the number ids of the units that completed the BTS measurement
    _hotPlates: hp_list
        A list containing instances of the hotplates that control the temperature on each test unit
    _impedanceAnalyzer: ia.ImpedanceAnalyzer
        An instance to the impedance analyzer object
    _keithley: keithley.Keithley
        An instance to the keithley source-meter object
    _loggingLevels: dict
        A map of logging level strings to integers, as defined in the logging module
    _mainLogger: logging.Logger
        The logger to the class, used to handle any messages within the class.
    _measurementConfig: config.ConfigParser
        An instance of ConfigParser that contains all the acquisition parameters
    _physicalTestUnits: int
        The number of hardware units available for use.
    _resourceManager:pyvisa.highlevel.ResourceManager
        An instance of pyvisa's resource manager to instantiate the instruments from.
    _scheduler: BackgroundScheduler
        An instance of the BackgroundScheduler
    _schedulerRunning: bool
        True if the scheduler is running, false otherwise
    _testUnits: int
        The number of test units configured to use in the BTS measurement

    Methods
    -------
    ramp_up(self, test_unit: int):
        Set's the hotplate temperature to the stress temperature. And sets the BTS flag to 'heating_up'. If the unit's
        fan is currently on, turns it off.

    ramp_down(self, test_unit: int):
        Set the hotplate temperature to 25 °C and turns the unit's fan on if it is off. Sets the BTS flag to
        'cooling_down'. Disconnects all the pins in the test unit.

    start_temperature_log(self, test_unit: int):
        Starts the temperature log. Instructs the scheduler to call the method '_log_temperature' according to the
        configured setting. Starts the class scheduler if it is not already running. Sets the '_schedulerRunning' to
        True.

    start_bts(self, test_unit: int):
        Starts the bias-temperature stress. This method should be called once the test unit has reached the stress
        temperature. It turn's on the voltage (connects all pins on the unit to the voltage source and turns the source
        on if it is not on already). Set the BTS flag to 'running_stress'. Instructs the scheduler to call the method
        'stop_bts' at time 1 stress_interval unit time later than the current time.

    stop_bts(self, test_unit: int):
        Calls 'ramp_down' method.

    _log_temperature(self, test_unit: int):
        Measures the temperature and leakage current through all the devices connected in parallel to the keithley
        source meter and saves the log to each of the device's datastores.
    """
    _activeCVUnit = []
    _arduinos: ard_list = []
    _btsAcquisitions: bts_list = []
    _cvSweepParams: dict = None
    _data_paths: List[str] = []
    _dlcpParams: dict = None
    _finishedUnits: List[int] = []
    _hotPlates: hp_list = []
    _impedanceAnalyzer: ia.ImpedanceAnalyzer = None
    _keithley: keithley.Keithley = None
    _loggingLevels = {'NOTSET': logging.NOTSET,
                      'DEBUG': logging.DEBUG,
                      'INFO': logging.INFO,
                      'WARNING': logging.WARNING,
                      'ERROR': logging.ERROR,
                      'CRITICAL': logging.CRITICAL}
    _mainLogger: logging.Logger = None
    _measurementConfig: configparser.ConfigParser = None
    _schedulerRunning = False
    _testUnits: int = 0

    def __init__(self, config_file_url: str, **kwargs):
        if not isinstance(config_file_url, str):
            raise TypeError('The first argument should be an instance of str.')
        self.debug: bool = kwargs.get('debug', False)
        system_option_requirements_json = kwargs.get('system_option_requirements_json',
                                                     'system_config_required_options.json')
        measurement_option_requirements_json = kwargs.get('measurement_options_requirements_json',
                                                          'measurement_config_required_options.json')

        # Load validation rules for the system configuration file
        self._configSystemRequiredOptions = self._read_json_file(system_option_requirements_json)
        # Load validation rules for the measurement configuration file
        self._configMeasurementRequiredOptions = self._read_json_file(measurement_option_requirements_json)

        # Load the system configuration file
        config = configparser.ConfigParser()
        config.read(config_file_url)

        self._physicalTestUnits = config.getint(section='global', option='test_units')

        # If the system configuration file is valid, then store it in the object
        if self._validate_config(config, self._configSystemRequiredOptions):
            self._systemConfig = config

        self._resourceManager: pyvisa.highlevel.ResourceManager = pyvisa.highlevel.ResourceManager()
        self._availableResources = self._resourceManager.list_resources()
        self._scheduler: BackgroundScheduler = BackgroundScheduler()

    def ramp_up(self, test_unit: int):
        bts_acquisition: bts.BTS = self._btsAcquisitions[test_unit]
        hp: hotplate.Hotplate = self._hotPlates[test_unit]
        hp.set_temperature(int(bts_acquisition.temperature))
        a: arduino_board.ArduinoBoard = self._arduinos[test_unit]
        if a.fan_status:
            a.fan_off()
        a.disconnect_all_pins()
        bts_acquisition.status = 'heating_up'

    def ramp_down(self, test_unit: int):
        hp: hotplate.Hotplate = self._hotPlates[test_unit]
        hp.set_temperature(25)
        a: arduino_board.ArduinoBoard = self._arduinos[test_unit]
        if not a.fan_status:
            a.fan_on()
        a.disconnect_all_pins()
        bts_acquisition: bts.BTS = self._btsAcquisitions[test_unit]
        bts_acquisition.status = 'cooling_down'

    def start_temperature_log(self, test_unit: int):
        bts_acquisition: bts.BTS = self._btsAcquisitions[test_unit]
        self._scheduler.add_job(func=self._log_temperature, trigger='interval', args=[test_unit],
                                seconds=bts_acquisition.temperature_sampling_interval,
                                id='temperature_log_unit{0}'.format(test_unit))
        if not self._schedulerRunning:
            self._scheduler.start()
            self._schedulerRunning = True

    def start_bts(self, test_unit: int):
        bts_acquisition: bts.BTS = self._btsAcquisitions[test_unit]
        a: arduino_board.ArduinoBoard = self._arduinos[test_unit]
        a.connect_all_pins()
        a.connect_keithley()
        # If the voltage source is off, turn it on
        if not self._keithley.source_on:
            self._keithley.turn_source_on()
        now: datetime = datetime.now()
        later = now + timedelta(seconds=bts_acquisition.stress_interval)
        self._scheduler.add_job(func=self.stop_bts, trigger='date', args=[test_unit],
                                date=later, id='bts_unit{0}'.format(test_unit))
        bts_acquisition.status = 'running_stress'
        bts_acquisition.start_bts_interval = now
        if not self._schedulerRunning:
            self._scheduler.start()
            self._schedulerRunning = True

    def stop_bts(self, test_unit: int):
        # Maybe we need to do something else here, else change all method calls to just, ramp_down.
        self.ramp_down(test_unit=test_unit)

    def _log_temperature(self, test_unit: int):
        a: arduino_board.ArduinoBoard = self._arduinos[test_unit]
        now = datetime.now()
        temperature = a.temperature
        current = self._keithley.current
        bts_acquisition: bts.BTS = self._btsAcquisitions[test_unit]
        clean_devices = bts_acquisition.clean_devices
        contaminated_devices = bts_acquisition.contaminated_devices
        dt = bts_acquisition.time_delta_total(current_datetime=now)
        for device in clean_devices:
            h5_storage: datastorage.H5Store = bts_acquisition.get_device_storage(device=device, clean=True)
            h5_storage.log_temperature_current(time=dt,
                                               temperature=temperature,
                                               current=current)

        for device in contaminated_devices:
            h5_storage: datastorage.H5Store = bts_acquisition.get_device_storage(device=device, clean=False)
            h5_storage.log_temperature_current(time=dt,
                                               temperature=temperature,
                                               current=current)

    def start_measurement(self):
        now = datetime.now()
        for test_unit in range(self._testUnits):
            self._log_cv_sweep(test_unit=test_unit)
            status_job_id = 'check_status{0}'.format(test_unit)
            self._scheduler.add_job(func=self._check_status, trigger='interval', args=[test_unit], id=status_job_id,
                                    seconds=10)
            self.start_temperature_log(test_unit=test_unit)

        if not self._schedulerRunning:
            self._scheduler.start()
            self._schedulerRunning = True

    def _check_status(self, test_unit: int):
        a: arduino_board.ArduinoBoard = self._arduinos[test_unit]
        bts_acquisition: bts.BTS = self._btsAcquisitions[test_unit]
        now = datetime.now()
        temperature = a.temperature
        # If the device temperature equals the target temperature and the status is ramping up, start stress
        if np.isclose(a.temperature, bts_acquisition.temperature,
                      atol=2) and bts_acquisition.status == 'heating_up':
            self.start_bts(test_unit=test_unit)
        elif a.temperature <= 26 and bts_acquisition.status == 'cooling_down':
            dt = bts_acquisition.time_delta_bts(current_datetime=datetime.now())
            bts_acquisition.accumulate_interval(dt=dt)
            bts_acquisition.status = "idle"
            self._log_cv_sweep(test_unit=test_unit)
        elif bts_acquisition.status == 'finished':
            self._unit_finished(test_unit=test_unit)

    def _unit_finished(self, test_unit: int):
        a: arduino_board.ArduinoBoard = self._arduinos[test_unit]
        if (a.temperature > 30) and (not a.fan_status):
            a.fan_on()
        elif a.fan_status:
            a.fan_off()

        # Stop logging temperature
        self._scheduler.remove_job(job_id='temperature_log_unit{0}'.format(test_unit))
        status_job_id = 'check_status{0}'.format(test_unit)
        # Stop checking if the unit is ready
        self._scheduler.remove_job(job_id=status_job_id)
        if test_unit not in self._finishedUnits:
            self._finishedUnits.append(test_unit)
        if len(self._finishedUnits) == self._testUnits:
            self._scheduler.shutdown()
            self._schedulerRunning = False

    def _log_cv_sweep(self, test_unit: int):
        # Make sure the board is idle
        bts_acquisition: bts.BTS = self._btsAcquisitions[test_unit]
        # If the current callback is in a scheduler queue, remove the job from the scheduler
        job_id = 'bts_cv_test_unit{0}'.format(test_unit)
        if job_id in self._scheduler.get_jobs():
            self._scheduler.remove_job(job_id=job_id)
        # Check if the measurement is not finished
        if bts_acquisition.status == "idle":
            # If the impedance analyzer is busy, schedule this measurement for later
            if len(self._activeCVUnit) > 0 or self._impedanceAnalyzer.status == "running":
                next_date = datetime.now() + timedelta(seconds=self._impedanceAnalyzer.wait_time)
                self._scheduler.add_job(self._log_cv_sweep(test_unit=test_unit),
                                        trigger='date', run_date=next_date,
                                        args=[test_unit], id=job_id)
            else:
                bts_acquisition.status = "running_cv"
                # Lock the Impedance Analyzer measurement to the test_unit
                self._activeCVUnit.append(test_unit)
                # Get the arduino for the selected board
                a: arduino_board.ArduinoBoard = self._arduinos[test_unit]
                if a.keithley_connected:
                    a.disconnect_keithley()
                if a.fan_status:  # If fan is on
                    a.fan_off()
                # Make sure no other pins are connected to the impedance analyzer
                for i in range(self._testUnits):
                    ai: arduino_board.ArduinoBoard = self._arduinos[i]
                    b: bts.BTS = self._btsAcquisitions[i]
                    if b.status != 'running_stress':
                        ai.disconnect_all_pins()
                # loop over all clean pins
                for d, p in zip(bts_acquisition.clean_devices, bts_acquisition.clean_pins):
                    # turn the pin on
                    a.pin_on(pin_number=p)
                    # collect the data from the impedance analyzer
                    data: vcr_type = self._impedanceAnalyzer.cv_sweep(
                        voltage_start=float(self._cvSweepParams['voltage_start']),
                        voltage_step=float(self._cvSweepParams['voltage_step']),
                        voltage_stop=float(self._cvSweepParams['voltage_stop']),
                        frequency=float(self._cvSweepParams['frequency']),
                        integration_time=self._cvSweepParams['integration_time'],
                        noa=int(self._cvSweepParams['number_of_averages']),
                        osc_amplitude=float(self._cvSweepParams['osc_amplitude']),
                        sweep_direction=self._cvSweepParams['sweep_direction']
                    )
                    # turn the pin off
                    a.pin_off(pin_number=p)
                    # get the storage for the pin
                    h5_ds = bts_acquisition.get_device_storage(device=d, clean=True)
                    # append the data to the h5 storage
                    h5_ds.append_cv(time=bts_acquisition.accumulated_stress_time, cv_data=data)
                # loop over al contaminated pins
                for d, p in zip(bts_acquisition.contaminated_devices, bts_acquisition.contaminated_pins):
                    a.pin_on(pin_number=p)
                    # collect the data from the impedance analyzer
                    data: vcr_type = self._impedanceAnalyzer.cv_sweep(
                        voltage_start=float(self._cvSweepParams['voltage_start']),
                        voltage_step=float(self._cvSweepParams['voltage_step']),
                        voltage_stop=float(self._cvSweepParams['voltage_stop']),
                        frequency=float(self._cvSweepParams['frequency']),
                        integration_time=self._cvSweepParams['integration_time'],
                        noa=int(self._cvSweepParams['number_of_averages']),
                        osc_amplitude=float(self._cvSweepParams['osc_amplitude']),
                        sweep_direction=self._cvSweepParams['sweep_direction']
                    )
                    # turn the pin off
                    a.pin_off(pin_number=p)
                    # get the storage for the pin
                    h5_ds = bts_acquisition.get_device_storage(device=d, clean=False)
                    # append the data to the h5 storage
                    h5_ds.append_cv(time=bts_acquisition.accumulated_stress_time, cv_data=data)

                # Unlock the impedance analyzer
                self._activeCVUnit.remove(test_unit)
                # Run a DLCP
                self._log_dlcp(test_unit=test_unit)
                if bts_acquisition.status != 'finished':
                    self.ramp_up(test_unit=test_unit)
                else:
                    self._unit_finished(test_unit=test_unit)

    def _log_dlcp(self, test_unit: int):
        # Make sure the board is idle
        bts_acquisition: bts.BTS = self._btsAcquisitions[test_unit]
        # If exists remove the job from the scheduler
        scheduled_jobs = self._scheduler.get_jobs()
        job_id = 'bts_dlcp_test_unit{0}'.format(test_unit)
        if job_id in scheduled_jobs:
            self._scheduler.remove_job(job_id=job_id)
        # Check if the measurement is not finished
        if bts_acquisition.status == "idle":
            # Check if there is something else being measured
            if len(self._activeCVUnit) > 0 or self._impedanceAnalyzer.status == "running":
                next_date = datetime.now() + timedelta(seconds=self._impedanceAnalyzer.wait_time)
                self._scheduler.add_job(self._log_dlcp(test_unit=test_unit),
                                        trigger='date', run_date=next_date,
                                        args=[test_unit], id=job_id)
            else:
                bts_acquisition.status = "running_dlcp"
                # Lock the Impedance Analyzer measurement
                self._activeCVUnit.append(test_unit)
                # Get the arduino for the selected board
                a: arduino_board.ArduinoBoard = self._arduinos[test_unit]
                # Make sure no other pins are connected to the impedance analyzer
                for i in range(self._testUnits):
                    ai: arduino_board.ArduinoBoard = self._arduinos[i]
                    b: bts.BTS = self._btsAcquisitions[i]
                    if b.status != 'running_stress':
                        ai.disconnect_all_pins()
                # Lock the impedance analyzer to this unit
                self._activeCVUnit.remove(test_unit)
                # loop over all clean pins
                for d, p in zip(bts_acquisition.clean_devices, bts_acquisition.clean_pins):
                    # turn the pin on
                    a.pin_on(pin_number=p)
                    # collect the data from the impedance analyzer
                    data: dlcp_type = self._impedanceAnalyzer.dlcp_sweep(
                        nominal_bias=float(self._dlcpParams['nominal_bias']),
                        start_amplitude=float(self._dlcpParams['start_amplitude']),
                        step_amplitude=float(self._dlcpParams['step_amplitude']),
                        stop_amplitude=float(self._dlcpParams['stop_amplitude']),
                        frequency=float(self._cvSweepParams['frequency']),
                        integration_time=self._cvSweepParams['integration_time'],
                        noa=int(self._cvSweepParams['number_of_averages']),
                    )
                    # turn the pin off
                    a.pin_off(pin_number=p)
                    # get the storage for the pin
                    h5_ds = bts_acquisition.get_device_storage(device=d, clean=True)
                    # append the data to the h5 storage
                    h5_ds.append_dlcp(time=bts_acquisition.accumulated_stress_time, dlcp_data=data)
                # loop over al contaminated pins
                for d, p in zip(bts_acquisition.contaminated_devices, bts_acquisition.contaminated_pins):
                    a.pin_on(pin_number=p)
                    # collect the data from the impedance analyzer
                    data: dlcp_type = self._impedanceAnalyzer.dlcp_sweep(
                        nominal_bias=float(self._dlcpParams['nominal_bias']),
                        start_amplitude=float(self._dlcpParams['start_amplitude']),
                        step_amplitude=float(self._dlcpParams['step_amplitude']),
                        stop_amplitude=float(self._dlcpParams['stop_amplitude']),
                        frequency=float(self._cvSweepParams['frequency']),
                        integration_time=self._cvSweepParams['integration_time'],
                        noa=int(self._cvSweepParams['number_of_averages']),
                    )
                    # turn the pin off
                    a.pin_off(pin_number=p)
                    # get the storage for the pin
                    h5_ds = bts_acquisition.get_device_storage(device=d, clean=False)
                    # append the data to the h5 storage
                    h5_ds.append_dlcp(time=bts_acquisition.accumulated_stress_time, dlcp_data=data)
                bts_acquisition.status = "idle"

    def load_test_config(self, config: configparser.ConfigParser):
        if not isinstance(config, configparser.ConfigParser):
            raise TypeError('The argument should be an instance of configparser.ConfigParser.')
        if self.debug:
            self._print('Loading measurement configuration...')  # No logger yet...

        if self._validate_config(config, self._configMeasurementRequiredOptions):
            all_sections = config.sections()
            test_units = len([u for u in all_sections if 'bts' in u])
            if test_units > self._physicalTestUnits:
                msg = 'Measurement requesting {0:d} units. Available units: {1}'.format(test_units,
                                                                                        self._physicalTestUnits)
                raise errors.ConfigurationError(message=msg)
            self._testUnits = test_units
            self._measurementConfig = config
            now = datetime.now()
            time_stamp = now.strftime('%Y%m%d')

            self._cvSweepParams = dict(config.items('cv_sweep'))
            self._dlcpParams = dict(config.items('dlcp'))

            # Create a directory structure by
            # +-- Temperature + Applied Electric Field
            #       +-- Device 1 - Clean
            #       +-- Device 2 - Clean
            #       | ...
            #       +-- Device 7 - Contaminated
            #       +-- Device 8 - Contaminated
            root_path: str = config.get(section='general', option='base_path')
            if platform.system() == 'Windows':
                root_path = '\\\\?\\' + root_path
            base_path = os.path.join(root_path, time_stamp)
            if self.debug:
                self._print('Creating base path at {0}'.format(base_path))  # No logger yet...
            self._create_path(base_path)
            # Create main logger
            self._mainLogger = self._create_logger(base_path, name='Main Logger', level='CRITICAL', console=True)

            for i in range(self._testUnits):
                section_name = 'bts{0}'.format(i)
                params = self._get_bts_metadata(section_name=section_name)
                # Create a BTS acquisition object
                bts_acquisition = bts.BTS(temperature=params['stress_temperature'], bias=params['stress_bias'],
                                          stress_interval=params['stress_interval'],
                                          temperature_sampling_interval=params['temperature_sampling_interval'],
                                          max_time=params['max_stress_time'])

                data_folder = '{0:.0f}C-{1:.1f}'.format(params['stress_temperature'], params['stress_bias'])
                relative_path = os.path.join(base_path, data_folder)
                self._data_paths.append(relative_path)
                self._create_path(relative_path)
                # Create separate h5 files for each pin
                clean_devices = config.get(section=section_name, option='clean_devices').split(',')
                clean_pins = config.get(section=section_name, option='clean_pins').split(',')
                contaminated_devices = config.get(section=section_name, option='contaminated_devices').split(',')
                contaminated_pins = config.get(section=section_name, option='contaminated_pins').split(',')
                for d, p in zip(clean_devices, clean_pins):
                    h5_name = os.path.join(relative_path, '{0}_{1}-clean.h5'.format(params['clean_wafer_id'], d))
                    ds = datastorage.H5Store(h5_name)
                    metadata = params
                    metadata['device_id'] = d
                    ds.metadata(metadata=metadata, group="/bts")
                    ds.metadata(metadata=self._cvSweepParams, group='/bts')
                    bts_acquisition.append_device(device=d, pin=p, h5_storage=ds, clean=True)

                for d, p in zip(contaminated_devices, contaminated_pins):
                    h5_name = os.path.join(relative_path,
                                           '{0}_{1}-contaminated.h5'.format(params['contaminated_wafer_id'], d))
                    ds = datastorage.H5Store(h5_name)
                    metadata = params
                    metadata['device_id'] = d
                    ds.metadata(metadata=metadata, group="/bts")
                    bts_acquisition.append_device(device=d, pin=p, h5_storage=ds, clean=False)

                self._btsAcquisitions.append(bts_acquisition)
            self._print('Loaded acquisition parameters successfully.', level='INFO')

    def _get_bts_metadata(self, section_name: str):
        if not isinstance(section_name, str):
            raise TypeError('The argument should be an instance of str.')
        config: configparser.ConfigParser = self._measurementConfig
        options = dict(config.items(section=section_name))
        options['stress_voltage'] = float(options['stress_voltage'])
        options['stress_temperature'] = float(options['stress_temperature'])
        options['max_stress_time'] = float(options['max_stress_time'])
        options['stress_interval'] = float(options['stress_interval'])
        options['temperature_sampling_interval'] = float(options['temperature_sampling_interval'])
        options['thickness'] = float(options['thickness'])

        return options

    def _validate_config(self, config: configparser.ConfigParser, required_options) -> bool:
        if not isinstance(config, configparser.ConfigParser):
            raise TypeError('The configuration argument must be an instance of configparser.ConfigParser.')
        for section in required_options:
            if section == 'arduino' or section == 'hotplate' or section == "bts":
                n = self._testUnits + 1
                section_units = ['{0}{1:d}'.format(section, i) for i in range(1, n)]
                for s in section_units:
                    if config.has_section(s):
                        for o in required_options[section]:
                            if not config.has_option(s, o):
                                msg = 'Config file must have option \'{0}\' for section \'{1}\''.format(o, s)
                                raise errors.SystemConfigError(message=msg, test_units=self._testUnits)
                    else:
                        msg = "Config file must have section '{0}'.".format(s)
                        raise errors.SystemConfigError(message=msg, test_units=self._testUnits)
            else:
                if config.has_section(section):
                    for o in required_options[section]:
                        if not config.has_option(section, o):
                            msg = 'Config file must have option \'{0}\' for section \'{1}\''.format(o, section)
                            raise errors.SystemConfigError(message=msg, test_units=self._testUnits)
                else:
                    msg = "Config file must have section '{0}'.".format(section)
                    raise errors.SystemConfigError(message=msg, test_units=self._testUnits)
        return True

    def resource_available(self, resource_address: str):
        if resource_address not in self._availableResources:
            return False
        return True

    def init_impedance_analyzer(self):
        address = self._systemConfig.get(section='impedance_analyzer', option='address')
        name = self._systemConfig.get(section='impedance_analyzer', option='name')
        if not self.resource_available(resource_address=address):
            msg = 'The impedance analyzer is not available at address \'{0}\''.format(address)
            raise errors.InstrumentError(address=address, resource_name=name, message=msg)

        self._impedanceAnalyzer = ia.ImpedanceAnalyzer(address=address, resource_manager=self._resourceManager)

    def init_keithley(self):
        address = self._systemConfig.get(section='keithley', option='address')
        name = self._systemConfig.get(section='keithley', option='name')
        if not self.resource_available(resource_address=address):
            msg = 'The Keithley source-meter is not available at address \'{0}\''.format(address)
            raise errors.InstrumentError(address=address, resource_name=name, message=msg)
        self._keithley = keithley.Keithley(address=address, resource_manager=self._resourceManager)

    def init_arduino_boards(self):
        for i in range(self._testUnits):
            section_name = 'arduino{0:d}'.format(i)
            address = self._systemConfig.get(section=section_name, option='address')
            name = self._systemConfig.get(section=section_name, option='name')
            all_options = dict(self._systemConfig.items(section=section_name))
            pin_mappings = {
                'keithley': all_options['pin_keithley'],
                'fan': all_options['pin_fan']
            }
            import re
            pattern = re.compile(r'pin(\d+)')
            for o in all_options:
                match = re.match(pattern=pattern, string=o)
                if match:
                    pin = int(match.groups()[0])
                    pin_mappings[pin] = all_options[o]
            if self.resource_available(resource_address=address):
                self._arduinos.append(arduino_board.ArduinoBoard(address=address, name=name,
                                                                 pin_mappings=pin_mappings))
            else:
                msg = 'The arduino board \'{0}\' was not found in address: \'{1}\''.format(name, address)
                raise errors.InstrumentError(address=address, resource_name=name, message=msg)

    def init_hotplates(self, calibration_file: str):
        calibration = configparser.ConfigParser()
        calibration.read(calibration_file)

        for i in range(self._testUnits):
            section_name = 'hotplate{0:d}'.format(i)
            address: str = self._systemConfig.get(section=section_name, option='address')
            name: str = self._systemConfig.get(section=section_name, option='name')
            if self.resource_available(resource_address=address):
                hp = hotplate.Hotplate(address=address, name=name)
                hp.load_calibration(calibration)
                self._hotPlates.append(hp)
            else:
                msg = "The hotplate '{0}' was not found in address: '{1}'".format(name, address)
                raise errors.InstrumentError(address=address, resource_name=name, message=msg)

    @staticmethod
    def _read_json_file(filename: str):
        file = open(filename, 'r')
        json_data = json.load(file)
        file.close()
        return json_data

    @staticmethod
    def _create_path(path: str):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def _create_logger(path: str, name: str = 'experiment_logger',
                       level: [str, int] = 'CRITICAL', console: bool = False) -> logging.Logger:
        experiment_logger = logging.getLogger(name)
        experiment_logger.setLevel(level)
        filename = 'progress.log'
        log_file = os.path.join(path, filename)
        # create file handler which logs even critical messages
        fh = logging.FileHandler(log_file)
        fh.setLevel(level=level)
        experiment_logger.addHandler(fh)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        if console:
            ch = logging.StreamHandler()
            ch.setLevel(level=level)
            ch.setFormatter(formatter)
            experiment_logger.addHandler(ch)

        return experiment_logger

    def _print(self, msg: str, level="INFO"):
        level_no = self._loggingLevels[level]
        if self._mainLogger is None:
            print(msg)
        elif isinstance(self._mainLogger, logging.Logger):

            self._mainLogger.log(level_no, msg)
