"""
This class provides remote functionality for a SCILOGEX hotplate.

@author Erick Martinez Loran <erickrmartinez@gmail.com>
"""

import serial
import numpy as np
from pydlcp import errors
import time
import logging
import configparser


class Hotplate:
    _calibration = {'a1': 0.00112759470035273,
                    'a2': 0.820085915455346,
                    'b': 11.0122612663442}
    _configRequiredOptions = ['a1', 'a2', 'b']
    _heatOn = False
    _hotplate: serial.Serial = None
    _hotplateConnected = False
    _logger: logging.Logger = None
    _loggingLevels = {'NOTSET': logging.NOTSET,
                      'DEBUG': logging.DEBUG,
                      'INFO': logging.INFO,
                      'WARNING': logging.WARNING,
                      'ERROR': logging.ERROR,
                      'CRITICAL': logging.CRITICAL}
    _targetTemperature = 25
    _MAX_FAILED_CALLS = 20

    def __init__(self, address: str, name: str, **kwargs):
        """
        Arguments
        ---------
        :param address: The port to which the hotplate is connected
        :type address: str
        :param name: The name assigned to the hotplate
        :type name: str
        :param kwargs: keyword arguments
        """
        baudrate = kwargs.get('baudrage', 9600)
        bytesize = kwargs.get('bitsize', 8)
        stopbits = kwargs.get('stopbits', 1)
        timeout = kwargs.get('timeout', 0.5)
        self._debug = kwargs.get('debug', False)
        self._address = address
        self._hotplate = serial.Serial()
        self._hotplate.baudrate = baudrate
        self._hotplate.bytesize = bytesize
        self._hotplate.stopbits = stopbits
        self._hotplate.timeout = timeout
        self._name = name

    def connect(self):
        if not self._hotplateConnected:
            self._hotplate.open()
            self._hotplateConnected = True
        else:
            msg = "Hotplate '{0}' already open on address '{1}'.".format(self._name, self._address)
            self._print(msg=msg, level='WARNING')

    def disconnect(self):
        if self._hotplateConnected:
            self._hotplate.close()
            self._hotplateConnected = False
        else:
            msg: str = "Arduino board '{0}' already closed.".format(self._name)
            self._print(msg=msg, level='WARNING')

    @staticmethod
    def _checksum(query) -> np.uint8:
        return sum(query[1:]) % 256

    def _write_query(self, query):
        for q in query:
            b = np.uint8(q)
            self._hotplate.write(b)
            time.sleep(0.05)
        time.sleep(0.1)

    def get_temperature_setpoint(self, failed_calls: np.uint8 = 0) -> float:
        """
        :param failed_calls: The number of times the function has been called unsuccessfully (default = 0)
        :type failed_calls: np.uint8
        :return set_temperature: The temperature set point
        :rtype set_temperature: float
        
        From SCILOGEX
        Section 3.3 Get status
        
        Command:
        -------------------------------------------------------
        1 | 2 | 3 | 4 | 5 | 6
        -------------------------------------------------------
        0xfe | 0xA2 | NULL | NULL | NULL | Check sum
        -------------------------------------------------------
        Response:
        -------------------------------------------------------
        1    | 2    | 3, 4, 5, 6, 7, 8, 9, 10 | 11
        -------------------------------------------------------
        0xfd | 0xA2 | Parameter1... 8         | Check sum
         -------------------------------------------------------
        Parameter5: temp set(high)
        Parameter6: temp set(low)
        """

        # Prepare the query to the hotplate
        query = [254, 162, 0, 0]
        checksum = self._checksum(query)
        query.append(checksum)
        try:
            self._write_query(query)
            out = self._hotplate.read(11)
            self._hotplate.flush()
            if len([out]) > 0:
                # Get the value of the set temp HT and LT from the hotplate
                thl = out[6:7]
                # Transform the value into decimal
                val = 0
                n = len(thl)
                for i in range(n):
                    val += 256**(n-i-1)*thl[i]
                set_temperature = val / 10
            else:
                msg = 'Failed to read the temperature set point in {0} - {1}. '.format(self._name, self._address)
                failed_calls += 1
                if failed_calls <= self._MAX_FAILED_CALLS:
                    msg += 'Trying again... (Attempt {0}/{1})'.format(failed_calls, self._MAX_FAILED_CALLS)
                    time.sleep(0.1)
                    set_temperature = self.get_temperature_setpoint(failed_calls=failed_calls)
                    self._print(msg, level='WARNING')
                else:
                    msg += 'Exceeded allowable number of attempts for {0} - {1}.'.format(failed_calls,
                                                                                         self._MAX_FAILED_CALLS)
                    raise Warning(msg)
        except serial.SerialTimeoutException as e:
            msg = 'Failed to read the temperature set point in {0} - {1}. '.format(self._name, self._address)
            msg += e.strerror
            if failed_calls <= self._MAX_FAILED_CALLS:
                msg += 'Trying again... (Attempt {0}/{1})'.format(failed_calls, self._MAX_FAILED_CALLS)
                self._print(msg, level='WARNING')
                time.sleep(0.1)
                set_temperature = self.get_temperature_setpoint(failed_calls=failed_calls)
            else:
                msg += 'Exceeded allowable number of attempts for {0} - {1}.'.format(failed_calls,
                                                                                     self._MAX_FAILED_CALLS)
                self._print(msg, level='ERROR')
                raise e
        return set_temperature

    def get_heating_status(self, failed_calls: np.uint8 = 0) -> bool:
        """
        :param failed_calls: The number of times the function has been called unsuccessfully (default = 0)
        :type failed_calls: np.uint8
        :return status: True if not heating False if heating
        :rtype status: bool

        % From SCILOGEX
        % Section 3.2 Get information
        %
        % Command:
        % -------------------------------------------------------
        %  1   | 2    | 3    | 4    | 5    | 6
        % -------------------------------------------------------
        % 0xfe | 0xA1 | NULL | NULL | NULL | Check sum
        % -------------------------------------------------------
        % Response:
        % -------------------------------------------------------
        %  1   | 2    | 3,4,5,6,7,8,9,10 | 11
        % -------------------------------------------------------
        % 0xfd | 0xA1 | Parameter1... 8  | Check sum
        % -------------------------------------------------------
        % Parameter3: temperature status (0: closed, 1: open)
        """
        # Prepare the query to the hotplate
        query = [254, 161, 0, 0]
        checksum = self._checksum(query)
        query.append(checksum)
        try:
            self._write_query(query)
            out = self._hotplate.read(11)
            self._hotplate.flush()
            if len([out]) > 0:
                status: bool = bool(out[4])
            else:
                msg = 'Failed to read the status in {0} - {1}. '.format(self._name, self._address)
                failed_calls += 1
                if failed_calls <= self._MAX_FAILED_CALLS:
                    msg += 'Trying again... (Attempt {0}/{1})'.format(failed_calls, self._MAX_FAILED_CALLS)
                    self._print(msg, level='WARNING')
                    return self.get_heating_status(failed_calls=failed_calls)
                else:
                    msg += 'Exceeded allowable number of attempts for {0} - {1}.'.format(failed_calls,
                                                                                         self._MAX_FAILED_CALLS)
                    raise Warning(msg)
        except serial.SerialTimeoutException as e:
            msg = 'Failed to read the temperature set point in {0} - {1}. '.format(self._name, self._address)
            msg += e.strerror
            if failed_calls <= self._MAX_FAILED_CALLS:
                msg += 'Trying again... (Attempt {0}/{1})'.format(failed_calls, self._MAX_FAILED_CALLS)
                self._print(msg, level='WARNING')
                time.sleep(0.1)
                return self.get_heating_status(failed_calls=failed_calls)
            else:
                msg += 'Exceeded allowable number of attempts for {0} - {1}.'.format(failed_calls,
                                                                                     self._MAX_FAILED_CALLS)
                self._print(msg, level='ERROR')
                raise e
        return status

    def set_temperature(self, temperature: int, failed_calls: np.uint8 = 0):
        current_setpoint = self.get_temperature_setpoint()
        corrected_temperature = self.correct_temperature_setpoint(temperature)
        if current_setpoint != temperature or self.get_heating_status():
            # Need to multiply by 10 to get the right temp setpoint
            set_temp = corrected_temperature / 10
            ht = np.uint8(np.floor(set_temp/256))
            lt = np.uint8(set_temp % 256)
            query = [254, 178, ht, lt, 0]
            checksum = self._checksum(query)
            query.append(checksum)
            try:
                self._write_query(query)
                out = self._hotplate.read(6)
                self._hotplate.flush()
                if len([out]) == 0 or out[2] == 1:
                    msg = r'Failed to set the temperature to {0:d} °C on {1} - {2}. '.format(temperature,
                                                                                             self._name, self._address)
                    failed_calls += 1
                    if failed_calls <= self._MAX_FAILED_CALLS:
                        msg += 'Trying again... (Attempt {0}/{1})'.format(failed_calls, self._MAX_FAILED_CALLS)
                        self._print(msg, level='WARNING')
                        self.set_temperature(temperature, failed_calls)
                    else:
                        msg += 'Exceeded allowed number of attempts for {0} - {1}'.format(self._name, self._address)
                        self._print(msg, level='ERROR')
                        e = errors.HotplateError(self._address, self._name, msg)
                        e.set_heating_status(self.get_heating_status())
                        e.set_temperature_setpoint(temperature)
                        raise e
            except serial.SerialTimeoutException as e:
                msg = r'Timeout error trying to set the temperature to {0:d} °C on {1} - {2}. '.format(temperature,
                                                                                                       self._name,
                                                                                                       self._address)
                failed_calls += 1
                if failed_calls <= self._MAX_FAILED_CALLS:
                    msg += 'Trying again... (Attempt {0}/{1})'.format(failed_calls, self._MAX_FAILED_CALLS)
                    self._print(msg, level='WARNING')
                    self.set_temperature(temperature, failed_calls)
                else:
                    msg += 'Exceeded allowed number of attempts for {0} - {1}'.format(self._name, self._address)
                    self._print(msg, level='ERROR')
                    raise e
            else:
                msg = 'Temperature set for {0} - {1} to {2:d}. Attempts {3}/{4}'.format(self._name,
                                                                                        self._address,
                                                                                        temperature,
                                                                                        failed_calls + 1,
                                                                                        self._MAX_FAILED_CALLS)
                self._print(msg)

    def set_calibration(self, a1: float, a2: float, b: float):
        """
        Arguments
        ----------
        :param a1: The coefficient to the second oder term
        :type a1: float
        :param a2: The coefficient to the first order term
        :type a2: float
        :param b: The zero order term
        :type b: float
        :return: None
        """
        self._calibration = {'a1': a1,
                             'a2': a2,
                             'b': b}

    def load_calibration(self, config: configparser.ConfigParser):
        """
        Attributes
        ----------
        :param config: The configuration parser
        :type config: configparser.ConfigParser
        :return: None
        """
        if self._validate_config(config, self._configRequiredOptions):
            self._calibration = {'a1': config.getfloat(self._name, 'a1'),
                                 'a2': config.getfloat(self._name, 'a2'),
                                 'b': config.getfloat(self._name, 'b')}

    def _validate_config(self, config: configparser.ConfigParser, required_options) -> bool:
        if not isinstance(config, configparser.ConfigParser):
            raise TypeError('The configuration argument must be an instance of configparser.ConfigParser.')
        if config.has_section(self._name):
            for o in required_options[self._name]:
                if not config.has_option(self._name, o):
                    msg = 'Config file must have option \'{0}\' for \'{1}\''.format(o, self._name)
                    raise errors.ConfigurationError(message=msg)
        return True

    def correct_temperature_setpoint(self, temperature: float) -> float:
        x = temperature
        a1: float = self._calibration['a1']
        a2: float = self._calibration['a2']
        b: float = self._calibration['b']
        return a1*x*x + a2*x + b

    def set_logger(self, logger: logging.Logger):
        self._logger = logger

    def _print(self, msg: str, level="INFO"):
        level_no = self._loggingLevels[level]
        if self._logger is None:
            print(msg)
        elif isinstance(self._logger, logging.Logger):

            self._logger.log(level_no, msg)

    def __del__(self):
        self.set_temperature(25)
        self.disconnect()
