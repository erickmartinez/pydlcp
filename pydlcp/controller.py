import numpy as np
import configparser
import json
from datetime import datetime
import os
import re
import logging
import pyvisa
from pydlcp import arduino_board, hotplate, errors, impedance_analyzer as ia, DLCPDataStore as dh5, bts
import platform
from typing import List

# Different data type definitions
from pydlcp.DLCPDataStore import DLCPDataStore

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
    This class provides methods to control a DLCP experiment and save the results to a h5 data store.

    """
    _dataPath: str = None
    _dlcpDataStore: dh5.DLCPDataStore = None
    _dlcpParams: dict = None
    _fileTag: str = None
    _hotPlates: hp_list = []
    _impedanceAnalyzer: ia.ImpedanceAnalyzer = None
    abort: bool = False
    _loggingLevels = {'NOTSET': logging.NOTSET,
                      'DEBUG': logging.DEBUG,
                      'INFO': logging.INFO,
                      'WARNING': logging.WARNING,
                      'ERROR': logging.ERROR,
                      'CRITICAL': logging.CRITICAL}
    _loggerName: str = None
    _measurementConfig: configparser.ConfigParser = None

    def __init__(self, config_file_url: str = None, **kwargs):
        cwd = os.path.join(os.getcwd(), 'pydlcp')
        if config_file_url is None:
            config_file_url = os.path.join(cwd, 'dlcp_hardware_config.ini')
        elif not isinstance(config_file_url, str):
            raise TypeError('The first argument should be an instance of str.')
        self.debug: bool = kwargs.get('debug', False)
        default_sys_required_options_json = os.path.join(cwd, 'dlcp_system_config_required_options.json')
        default_dlcp_meas_required_json = os.path.join(cwd, 'dlcp_measurement_config_required_options.json')
        system_option_requirements_json = kwargs.get('dlcp_system_option_requirements_json',
                                                     default_sys_required_options_json)
        measurement_option_requirements_json = kwargs.get('dlcp_measurement_options_requirements_json',
                                                          default_dlcp_meas_required_json)

        # Load validation rules for the system configuration file
        self._configSystemRequiredOptions = self._read_json_file(system_option_requirements_json)
        # Load validation rules for the measurement configuration file
        self._configMeasurementRequiredOptions = self._read_json_file(measurement_option_requirements_json)

        # Load the system configuration file
        config = configparser.ConfigParser()
        config.read(config_file_url)

        # If the system configuration file is valid, then store it in the object
        if self._validate_config(config, self._configSystemRequiredOptions):
            self._systemConfig = config

        self._resourceManager: pyvisa.highlevel.ResourceManager = pyvisa.highlevel.ResourceManager()
        self._availableResources = self._resourceManager.list_resources()

    def load_test_config(self, config: configparser.ConfigParser):
        """
        Load the acquisition settings. Follows

        1. Loads the configuration object
        2. Validates the configuration settings using the rules provided by the constructor (default rules are in
        ./dlcp_system_config_required_options.json).
        3. If valid, creates the data structure for the measurement.
        4. Creates a data storage object that will ouput the data to an h5 file.

        Parameters
        ----------
        config: configparser.ConfigParser
            The configuration settings as read from the ini file specified on the constructor

        Raises
        ------
        TypeError:
            If the config argument is not an instance of configparser.ConfigParser
        """
        if not isinstance(config, configparser.ConfigParser):
            raise TypeError('The argument should be an instance of configparser.ConfigParser.')
        if self.debug:
            self._print('Loading measurement configuration...')  # No logger yet...

        if self._validate_config(config, self._configMeasurementRequiredOptions):
            self._measurementConfig = config
            now = datetime.now()
            time_stamp = now.strftime('%Y%m%d')
            # iso_date = now.astimezone().isoformat()

            self._dlcpParams = dict(config.items('dlcp'))
            self._fileTag = config.get(section='general', option='file_tag')

            base_path: str = config.get(section='general', option='base_path')
            if platform.system() == 'Windows':
                base_path = r'\\?\\' + base_path

            base_path = self._create_path(base_path)
            # Create main logger
            self._loggerName: logging.Logger = self._create_logger(base_path, name='Main Logger',
                                                                   level='DEBUG', console=True)
            if self._impedanceAnalyzer is not None:
                self._impedanceAnalyzer.set_logger(logger=self._loggerName)

            if self.debug:
                self._print('Created base path at {0}'.format(base_path))  # No logger yet...

            self._dataPath = base_path

            self._print('Loaded acquisition parameters successfully.', level='INFO')
            h5_name = os.path.join(self._dataPath, '{0}_{1}.h5'.format(self._fileTag, time_stamp))
            ds: DLCPDataStore = dh5.DLCPDataStore(file_path=h5_name)
            metadata = self._dlcpParams
            metadata['file_tag'] = self._fileTag
            ds.metadata(metadata=metadata, group="/dlcp")
            self._dlcpDataStore = ds

    def start_dlcp(self) -> int:
        """
        Starts the DLCP acquisition.

        1. Loads the acquisition parameters from the _dlcpParams class property.
        2. Iterates over all the nominal biases
        3. Saves the data on the _dlcpDataStore

        Returns
        -------
        int:
            0 if the measurement was interrupted
            1 if it was successful.
        """
        ds = self._dlcpDataStore
        dlcp_params = self._dlcpParams
        circuit = dlcp_params['circuit']
        nb_start = float(dlcp_params['nominal_bias_start'])
        nb_step = float(dlcp_params['nominal_bias_step'])
        nb_stop = float(dlcp_params['nominal_bias_stop'])
        osc_level_start = float(dlcp_params['osc_level_start'])
        osc_level_step = float(dlcp_params['osc_level_step'])
        osc_level_stop = float(dlcp_params['osc_level_stop'])
        freq = float(dlcp_params['frequency'])
        integration_time = dlcp_params['integration_time']
        noa = int(dlcp_params['number_of_averages'])
        # Iterate from nominal bias start to nominal bias stop
        nb_scan = np.arange(start=nb_start, stop=nb_stop+nb_step, step=nb_step)
        for i, nb in enumerate(nb_scan):
            progress_str = 'Acquiring capacitance for Nominal Bias = {0:.3f} V'.format(nb)
            self._print(progress_str)
            data = self._impedanceAnalyzer.dlcp_sweep(nominal_bias=nb, osc_start=osc_level_start,
                                                      osc_step=osc_level_step, osc_stop=osc_level_stop,
                                                      frequency=freq, integration_time=integration_time,
                                                      noa=noa, circuit=circuit)
            ds.save_dlcp(dlcp_data=data, nominal_bias=nb)
            if self.abort:
                self.abort = False
                return 0
        return 1

    def cv_sweep(self, voltage_start: float, voltage_step: float, voltage_stop: float, frequency: float,
                 **kwargs):
        """
        Runs a capacitance-voltage sweep and saves it to the h5 data store.

        Parameters
        ----------
        voltage_start: float
            The start DC bias (V)
        voltage_step: float
            The step DC bias (V)
        voltage_stop: float
            The stop DC bias (V)
        frequency: float
            The AC frequency
        kwargs:
            keyword arguments passed to `impedance_analyzer.cv_sweep' method.
        """
        data = self._impedanceAnalyzer.cv_sweep(voltage_start=voltage_start,
                                                voltage_step=voltage_step,
                                                voltage_stop=voltage_stop,
                                                frequency=frequency, **kwargs)
        self._dlcpDataStore.save_cv(cv_data=data)
        sweep_params = {
            'voltage_start': voltage_start,
            'voltage_step': voltage_step,
            'voltage_stop': voltage_stop,
            'frequency': frequency
        }

        for k, v in kwargs.items():
            sweep_params[k] = v

        self._dlcpDataStore.metadata(metadata=sweep_params, group='/cv')
        return data

    def _validate_config(self, config: configparser.ConfigParser, required_options: dict) -> bool:
        """
        Validate the configuration based on an agreed set of rules

        Parameters
        ----------
        config: configparser.ConfigParser
            The configuration parsed from an .ini file
        required_options: dict
            A dictionary with the rules
        Returns
        -------
        bool
            True if the configuration meets the rules defined in required options. False otherwise
        """
        if not isinstance(config, configparser.ConfigParser):
            raise TypeError('The configuration argument must be an instance of configparser.ConfigParser.')
        for section in required_options:
            if config.has_section(section):
                for o in required_options[section]:
                    if not config.has_option(section, o):
                        msg = 'Config file must have option \'{0}\' for section \'{1}\''.format(o, section)
                        self._print(msg=msg, level="ERROR")
                        raise errors.ConfigurationError(message=msg)
            else:
                msg = "Config file must have section '{0}'.".format(section)
                raise errors.ConfigurationError(message=msg)
        return True

    def resource_available(self, resource_address: str):
        """
        Checks if the specified resource is available through the pyvisa connection

        Parameters
        ----------
        resource_address: str
            The address of the resource

        Returns
        -------
        bool
            True if the resource exists in the list of available resources, false otherwise.
        """
        if resource_address not in self._availableResources:
            return False
        return True

    def init_impedance_analyzer(self):
        """
        Tries to create the impedance analyzer pyvisa object and open the connection to it.

        Raises
        ------
        errors.InstrumentError:
            If the address to the pyvisa resource for the impedance analyzer is not available.
        """
        address = self._systemConfig.get(section='impedance_analyzer', option='address')
        name = self._systemConfig.get(section='impedance_analyzer', option='name')
        if not self.resource_available(resource_address=address):
            msg = 'The impedance analyzer is not available at address \'{0}\''.format(address)
            raise errors.InstrumentError(address=address, resource_name=name, message=msg)

        self._impedanceAnalyzer = ia.ImpedanceAnalyzer(address=address, resource_manager=self._resourceManager)
        if self._loggerName is not None:
            self._impedanceAnalyzer.set_logger(logger=self._loggerName)

    @staticmethod
    def _read_json_file(filename: str):
        """
        Reads the json file containing the rules to validate the configuration.

        Parameters
        ----------
        filename: str
            The full name to the json file containing the validation rules.

        Returns
        -------
        json:
            The json data in the file
        """
        file = open(filename, 'r')
        json_data = json.load(file)
        file.close()
        return json_data

    def _create_path(self, path: str, overwrite: bool = True):
        """
        Creates a folder in the selected path. If the folder exists and the overwrite flag is set to False, it appends
        a consecutive number to the path.

        Parameters
        ----------
        path: str
            The path to be created (if it does not exist)
        overwrite: bool
            If set to true and the folder exists, it will not try to create a folder. Otherwise, it will append a
            consecutive integer to te path and try to create it.

        Returns
        -------
        str:
            The newly created path
        """
        if not os.path.exists(path):
            os.makedirs(path)
            return path
        elif not overwrite:
            relative_name = os.path.basename(path)
            parent_dir = os.path.dirname(path)
            p = re.compile(r'{0}\d*'.format(relative_name))
            paths = [f for f in os.listdir(parent_dir) if p.match(f)]
            n_paths = len(paths)
            new_path = '{0}_{1:d}'.format(path, n_paths)
            self._create_path(path=new_path, overwrite=True)
            return new_path
        else:
            return path

    def _create_logger(self, path: str, name: str = 'experiment_logger',
                       level: [str, int] = 'DEBUG', console: bool = False) -> logging.Logger:
        """
        Creates an instance of logging.Logger saving the logs to the specified file.

        Parameters
        ----------
        path: str
            The folder to store the log file in
        name:str
            The name to identify the logger. Default: 'experiment_logger'.
        level:str
            The threshold level for the logs (default 'DEBUG')
        console: bool
            True if allowing output to the console as well.

        Returns
        -------
        logging.Logger:
            The logger instance

        """
        self._loggerName = name
        experiment_logger: logging.Logger = logging.getLogger(name)
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

    @property
    def impedance_analyzer_address(self) -> str:
        address = self._systemConfig.get(section='impedance_analyzer', option='address')
        return address

    @property
    def impedance_analyzer_resource_name(self) -> str:
        name = self._systemConfig.get(section='impedance_analyzer', option='name')
        return name

    def connect_devices(self):
        if self._impedanceAnalyzer is not None:
            self._impedanceAnalyzer.connect()
        else:
            self.init_impedance_analyzer()
            self._impedanceAnalyzer.connect()

    def disconnect_devices(self):
        if self._impedanceAnalyzer is not None:
            self._impedanceAnalyzer.disconnect()

    def __del__(self):
        try:
            self.disconnect_devices()
        except Exception as e:
            self._print(msg='Error disconnecting devices.', level='ERROR')
            self._print(e)
        finally:
            if self._loggerName is not None:
                # remove the log handlers
                experiment_logger: logging.Logger = logging.getLogger(self._loggerName)
                try:
                    handlers = experiment_logger.handlers[:]
                    for h in handlers:
                        h.close()
                        experiment_logger.removeHandler(h)
                except Exception as e:
                    print(e)
                self._loggerName = None

    def _print(self, msg: str, level="DEBUG"):
        """
        Handles the output messages for the class

        Parameters
        ----------
        msg: str
            The message to be output
        level: str
            The level of the message (DEBUG, INFO, ERROR, CRITICAL). Only used if a logger has been specified.
        """
        level_no = self._loggingLevels[level]
        if self._loggerName is None:
            print(msg)
        else:
            experiment_logger: logging.Logger = logging.getLogger(self._loggerName)
            experiment_logger.log(level_no, msg)
