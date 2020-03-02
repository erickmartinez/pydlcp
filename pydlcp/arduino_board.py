import logging
from string import Template
import subprocess
import serial
import numpy as np
import os
import platform
import time
import pydlcp.errors as errors
from typing import List


class ArduinoBoard:
    """
    This class provides basic control to an Arduino board running a serial controller

    ...

    Attributes
    ----------
    _activePins: List[int]
        The pins that are currently active
    _board: serial.Serial
        The serial connection to the arduino board.
    _boardConnected: bool
        True if the serial connection to the board is open, false otherwise
    _pinMappings: dict
        A dictionary containing the map of pins for different functionalities
    _fanOn: bool
        True if the fan is on false otherwise
    _keithleyConnected: bool
        True if all the measuring pins are connected to the keithley source meter, false if they are connected to the
        impedance analyzer

    Methods
    -------
    connect(self):
        Opens the serial connection to the arduino server

    disconnect(self):
        Closes the serial connection to the arduino server

    connect_keithley(self):
        Triggers the relay to connect all the measuring pins to the keithley source meter instead of having them
        connected to the impedance analyzer

    pin_on(self, pin_number: int):
        Triggers the relay to turn the specified pin on.

    pin_off(self, pin_number: int):
        Triggers the relay to turn the specified pin off.

    fan_on(self):
        Triggers the relay to turn the fan on.

    fan_off(self):
        Triggers the relay to turn the fan off.

    disconnect_all_pins(self):
        Disconnects all measuring pins.

    connect_all_pins(self):
        Connects all measuring pins.

    temperature(self):
        Reads the temperature from the MAX31855 thermocouple.

    set_logger(self, logger: logging.Logger):
        Sets the logger that will handle the workflow messages instead of python's print.

    create_serial_server(self):
        Creates an arduino sketch file containing a code to setup a serial server to controll the test unit. Then it
        verifies it and uploads it to the arduino board.

    _board_write(self, command: str):
        Sends a command to the arduino controller using serial communication

    _board_query(self, command: str) -> str:
        Queries the arduino controller and reads the output using serial communication

    _print(self, msg: str, level="INFO"):
        Prints a system message. If a logger is specified, handles the message with the logger, else it prints it to the
        console using 'print'.

    _create_path(path: str)
        Creates a path if it does not exist. Used to create the directory structure for the arduino sketches.

    """
    _activePins = []
    _arduinoCmd = r'C:/Program Files (x86)/Arduino/arduino.exe'
    _board: serial.Serial = None
    _boardConnected = False
    _pinMappings = {'keithley': 'A0',
                    'fan': 'A1',
                    'thermocouple': '10',
                    1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9'}
    _fanOn = False
    _keithleyConnected = True
    _logger: logging.Logger = None
    _loggingLevels = {'NOTSET': logging.NOTSET,
                      'DEBUG': logging.DEBUG,
                      'INFO': logging.INFO,
                      'WARNING': logging.WARNING,
                      'ERROR': logging.ERROR,
                      'CRITICAL': logging.CRITICAL}

    def __init__(self, address: str, name: str, pin_mappings: dict):
        """
        Parameters
        ----------
        address: str
            The port at which the board is accessed
        name: str
            The name of the board
        pin_mappings:
            The convened mapping for the pins
        """
        cwd = os.path.dirname(os.path.realpath(__file__))
        self._address = address
        self._name = name
        self._pinMappings = pin_mappings
        sketch_path = r'arduino_sketches\serial_controller_{0}'.format(name)
        self._create_path(os.path.join(cwd, sketch_path))
        server_sketch = os.path.join(sketch_path, 'serial_controller_{0}.ino'.format(name))
        server_template = r'arduino_sketches\arduino_serial_controller_template.txt'
        self._serverSketch = os.path.join(cwd, server_sketch)
        self._serverTemplate = os.path.join(cwd, server_template)
        if platform.system() == 'Windows':
            self._serverSketch = '\\\\?\\' + self._serverSketch
            self._serverTemplate = '\\\\?\\' + self._serverTemplate
        self.create_serial_server()
        self._board = serial.Serial()
        self._board.baudrate = 115200
        self._board.port = address

    def connect(self):
        """
        Opens the serial communication with the Arduino controller
        """
        if not self._boardConnected:
            self._board.open()
            self._boardConnected = True
        else:
            msg = "Arduino board '{0}' already open on address '{1}'.".format(self._name, self._address)
            self._print(msg=msg, level='WARNING')

    def disconnect(self):
        """
        Closes the serial connection with the Arduino controller
        """
        if self._boardConnected:
            self._activePins = []
            self.connect_keithley()
            self._board.close()
            self._boardConnected = False
        else:
            msg = "Arduino board '{0}' already closed.".format(self._name)
            self._print(msg=msg, level='WARNING')

    def connect_keithley(self):
        """
        Triggers the relay to connect the sample pins to the Keithley source-meter

        Warnings
        --------
        Warning
            If the Keithley source-meter was already connected.
        """
        if not self._keithleyConnected:
            q = r'TOGGLE PIN_KEITHLEY ON'
            self._board_write(q)
            self._keithleyConnected = True
            if 'keithley' in self._activePins:
                self._activePins.remove('keithley')
        else:
            msg = 'Keithley was already connected to {0} - {1}'.format(self._name, self._address)
            raise Warning(msg)

    def disconnect_keithley(self):
        """
        Triggers the relay to disconnect the sample pins to the Keithley source-meter.

        Warnings
        --------
        Warning
            If the Keithley source meter was already disconnected.
        """
        if self._keithleyConnected:
            q = r'TOGGLE PIN_KEITHLEY OFF'
            self._board_write(q)
            if 'keithley' not in self._activePins:
                self._activePins.append('keithley')
            self._keithleyConnected = False
        else:
            msg = 'Keithley was already disconnected from {0} - {1}'.format(self._name, self._address)
            raise Warning(msg)

    @property
    def keithley_connected(self) -> bool:
        return self._keithleyConnected

    def pin_on(self, pin_number: int):
        """
        Triggers the relay to turn the specified pin on
        Parameters
        ----------
        pin_number: int
            The number of the pin to turn on

        Raises
        ------
        errors.ArduinoError
            If the number of the pin is not within the pins defined in _pinMappings

        Warnings
        --------
        Warning
            If the pin was already connected.
        """
        if pin_number in self._pinMappings:
            pin = 'P{0:d}'.format(pin_number)
        else:
            message = 'Invalid pin number: \'{0}\'.'.format(pin_number)
            raise errors.ArduinoError(address=self._address, name=self._name, message=message)

        if pin not in self._activePins:
            q = r'TOGGLE {0} ON'.format(pin)
            self._board_write(q)
            self._activePins.append(pin)
        else:
            msg = 'Pin #{0} was already connected in {1} - {2}.'.format(pin_number, self._name, self._address)
            raise Warning(msg)

    def pin_off(self, pin_number: int):
        """
        Triggers the relay to turn the specified pin off
        Parameters
        ----------
        pin_number: int
            The number of the pin to turn on

        Raises
        ------
        errors.ArduinoError
            If the number of the pin is not within the pins defined in _pinMappings

        Warnings
        --------
        Warning
            If the pin was already disconnected.
        """
        if pin_number in self._pinMappings:
            pin = 'P{0:d}'.format(pin_number)  # self._pinMappings[pin_number]
        else:
            message = 'Invalid pin number: \'{0}\'.'.format(pin_number)
            raise errors.ArduinoError(address=self._address, name=self._name, message=message)

        if pin in self._activePins:
            q = r'TOGGLE {0} OFF'.format(pin)
            self._board_write(q)
            self._activePins.remove(pin)
        else:
            msg = 'Pin #{0} was already disconnected in {1} - {2}.'.format(pin_number, self._name, self._address)
            raise Warning(msg)

    def fan_on(self):
        """
        Triggers the relay to turn the fan on for the current board.

        Warnings
        ------
        Warning
            If the fan was already on.
        """
        if not self._fanOn:
            pin = 'PIN_FAN'
            q = r'TOGGLE {0} ON'.format(pin)
            self._board_write(q)
            self._fanOn = True
            if pin not in self._activePins:
                self._activePins.append(pin)
        else:
            msg = 'Fan was already connected to {0} - {1}'.format(self._name, self._address)
            raise Warning(msg)

    def fan_off(self):
        """
        Triggers the relay to turn the fan off.

        Warnings
        --------
        Warning
            If the fan was already off.
        """
        if self._fanOn:
            pin = 'PIN_FAN'
            q = r'TOGGLE {0} OFF'.format(pin)
            self._board_write(q)
            self._fanOn = False
            if pin not in self._activePins:
                self._activePins.append(pin)
        else:
            msg = 'Fan was already disconnected from {0} -  {1}'.format(self._name, self._address)
            raise Warning(msg)

    def disconnect_all_pins(self):
        """
        Disconnects all measuring pins from the instruments.
        """
        for p in range(1, 9):
            self.pin_off(p)

    def connect_all_pins(self):
        """
        Connects all measuring pins (by default, to the Impedance Analyzer).
        """
        for p in range(1, 9):
            self.pin_on(p)

    @property
    def fan_status(self) -> bool:
        return self._fanOn

    @property
    def temperature(self) -> float:
        q = 'TEMP'
        temperature = self._board_query(q)
        if temperature == 'nan':
            msg = 'Error reading temperature on {0} - {1}.'.format(self._name, self._address)
            self._print(msg=msg, level='WARNING')
            return np.nan
        else:
            return float(temperature)

    def set_logger(self, logger: logging.Logger):
        """
        Defines a logger for the class

        Parameters
        ----------
        logger: logger: logging.Logger
            A logger

        Raises
        ------
        TypeError
            If the provided logger is not an instance of logging.Logger.
        """
        if isinstance(logger, logging.Logger):
            self._logger = logger
        else:
            msg = 'Logger must be an instance of logging.Logger'
            raise TypeError(msg)

    def create_serial_server(self):
        """
        Creates an Arduino sketch that defines the logic of the Arduino board and creates a Serial controller to
        access the logic on the Arduino through a PC.

        The configuration of the pins on the sketch is injected from the
        pinMappings by substituting the right placeholders on a template file. It then generates a sketch folder with
        the .ino files. The method uses arduino.exe to verify and upload the sketch to the Arduino board in the selected
        port.

        Raises
        ------
        subprocess.CalledProcessError
            If there is a problem compiling or uplading the sketch to the board.
        """
        # Load the template file
        filein = open(self._serverTemplate, 'r')
        src = Template(filein.read())
        filein.close()
        # Substitute the pin mappings accordingly
        substitutions = {}
        for k, v in self._pinMappings.items():
            if isinstance(k, int):
                substitutions['p{0}'.format(k)] = v
            else:
                substitutions[k] = v
        out = src.safe_substitute(substitutions)
        # Write the output file
        fileout = open(self._serverSketch, 'w')
        fileout.write(out)
        fileout.close()
        # Compile and upload the file to the arduino on the selected address
        try:
            subprocess.check_call([self._arduinoCmd, '--verify', self._serverSketch, '--port', self._address,
                                   '--board', 'arduino:avr:uno'])
            time.sleep(1)
            subprocess.check_call([self._arduinoCmd, '--upload', self._serverSketch, '--port', self._address,
                                   '--board', 'arduino:avr:uno'])
            time.sleep(3)
        except subprocess.CalledProcessError as e:
            raise e
        else:
            msg = r"Controller sketch successfully compiled and uploaded to '{0}' - {1}.".format(self._name,
                                                                                                 self._address)
            self._print(msg=msg)

    def _board_write(self, command: str):
        """
        Sends a command to the arduino Serial controller

        Parameters
        ----------
        command: str
            The command to be sent. Options are "HELLO, TOGGLE [PIN] [ON|OFF], TEMP"
        """
        if not self._boardConnected:
            raise Warning('First make sure the board \'{0}\' is connected.'.format(self._name))
        q = '{0}\n'.format(command)
        self._board.write(q.encode('utf-8'))
        self._board.flush()

    def _board_query(self, command: str) -> str:
        """
        Sends a query to the arduino serial controller
        Current available queries are "TEMP"

        Parameters
        ----------
        command: str
            The command to be sent

        Returns
        -------
        str
            The response from the arduino controller

        Warnings
        --------
        Warning
            If the board is not connected
        """
        if not self._boardConnected:
            raise Warning('First make sure the board \'{0}\' is connected.'.format(self._name))
        self._board_write(command)
        output = self._board.readline()
        return output.decode('utf-8')

    def _print(self, msg: str, level="INFO"):
        """
        An output printer that either prints the messages to the console or handles them through the defined logger

        Parameters
        msg: str
            The message to be printed
        level: str
            The level of the message (only used if a logger is available)
        """
        level_no = self._loggingLevels[level]
        if self._logger is None:
            print(msg)
        elif isinstance(self._logger, logging.Logger):
            self._logger.log(level_no, msg)

    @staticmethod
    def _create_path(path: str):
        if not os.path.exists(path):
            os.makedirs(path)

    def __del__(self):
        self.disconnect()
        self._board = None
