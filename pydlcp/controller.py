import numpy as np
import configparser
import json
from datetime import datetime
import os
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
    _deviceName: str = None
    _dlcpDataStore: dh5.DLCPDataStore = None
    _dlcpParams: dict = None
    _hotPlates: hp_list = []
    _impedanceAnalyzer: ia.ImpedanceAnalyzer = None
    _loggingLevels = {'NOTSET': logging.NOTSET,
                      'DEBUG': logging.DEBUG,
                      'INFO': logging.INFO,
                      'WARNING': logging.WARNING,
                      'ERROR': logging.ERROR,
                      'CRITICAL': logging.CRITICAL}
    _mainLogger: logging.Logger = None
    _measurementConfig: configparser.ConfigParser = None

    def __init__(self, config_file_url: str, **kwargs):
        if not isinstance(config_file_url, str):
            raise TypeError('The first argument should be an instance of str.')
        self.debug: bool = kwargs.get('debug', False)
        system_option_requirements_json = kwargs.get('dlcp_system_option_requirements_json',
                                                     'dlcp_system_config_required_options.json')
        measurement_option_requirements_json = kwargs.get('dlcp_measurement_options_requirements_json',
                                                          'dlcp_measurement_config_required_options.json')

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
            self._deviceName = config.get(section='general', option='device_name')

            base_path: str = config.get(section='general', option='base_path')
            if platform.system() == 'Windows':
                base_path = '\\\\?\\' + base_path
            if os.path.exists(base_path):
                i = 0
                while os.path.exists("{0}_{1:d}".format(base_path, i)):
                    i += 1
                base_path = base_path + '_{0:d}'.format(i)

            if self.debug:
                self._print('Creating base path at {0}'.format(base_path))  # No logger yet...
            self._create_path(base_path)
            self._dataPath = base_path
            # Create main logger
            self._mainLogger = self._create_logger(base_path, name='Main Logger', level='CRITICAL', console=True)
            self._print('Loaded acquisition parameters successfully.', level='INFO')
            h5_name = os.path.join(self._dataPath, '{0}_{1}.h5'.format(self._deviceName, time_stamp))
            ds: DLCPDataStore = dh5.DLCPDataStore(file_path=h5_name)
            metadata = self._dlcpParams
            metadata['device_id'] = self._deviceName
            ds.metadata(metadata=metadata, group="/dlcp")
            self._dlcpDataStore = ds

    def start_dlcp(self):
        ds = self._dlcpDataStore
        dlcp_params = self._dlcpParams
        nb_start = float(dlcp_params['nominal_bias_start'])
        nb_step = float(dlcp_params['nominal_bias_step'])
        nb_stop = float(dlcp_params['nominal_bias_stop'])
        osc_level_start = float(dlcp_params['osc_level_start'])
        osc_level_step = float(dlcp_params['osc_level_step'])
        osc_level_stop = float(dlcp_params['osc_level_stop'])
        freq = float(dlcp_params['frequency'])
        integration_time = dlcp_params['integration_time']
        noa = int(dlcp_params['number_of_averages'])
        nb_scan = np.arange(start=nb_start, stop=nb_stop+nb_step, step=nb_step)
        for i, nb in enumerate(nb_scan):
            data = self._impedanceAnalyzer.dlcp_sweep(nominal_bias=nb, osc_start=osc_level_start,
                                                      osc_step=osc_level_step, osc_stop=osc_level_stop,
                                                      frequency=freq, integration_time=integration_time,
                                                      noa=noa)
            ds.save_dlcp(dlcp_data=data, nominal_bias=nb)

    def cv_sweep(self, voltage_start: float, voltage_step: float, voltage_stop: float, frequency: float,
                 **kwargs):
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

    @staticmethod
    def _validate_config(config: configparser.ConfigParser, required_options) -> bool:
        if not isinstance(config, configparser.ConfigParser):
            raise TypeError('The configuration argument must be an instance of configparser.ConfigParser.')
        for section in required_options:
            if config.has_section(section):
                for o in required_options[section]:
                    if not config.has_option(section, o):
                        msg = 'Config file must have option \'{0}\' for section \'{1}\''.format(o, section)
                        raise errors.DLCPSystemConfigError(message=msg)
            else:
                msg = "Config file must have section '{0}'.".format(section)
                raise errors.DLCPSystemConfigError(message=msg)
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
        self._impedanceAnalyzer.connect()

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
                       level: [str, int] = 'DEBUG', console: bool = False) -> logging.Logger:
        experiment_logger = logging.getLogger(name)
        experiment_logger.setLevel(level)
        filename = 'progress.log'
        log_file = os.path.join(path, filename)
        if platform.system() == 'Windows':
            log_file = "\\?\\" + log_file
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

    def _print(self, msg: str, level="DEBUG"):
        level_no = self._loggingLevels[level]
        if self._mainLogger is None:
            print(msg)
        elif isinstance(self._mainLogger, logging.Logger):

            self._mainLogger.log(level_no, msg)
