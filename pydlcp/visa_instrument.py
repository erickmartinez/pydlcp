"""
This class defines the minimum functionality for a visa instrument

@author: Erick Martinez Loran <erickrmartinez@gmail.com>
"""
import pyvisa
import numpy as np
from pydlcp import errors
import logging


class VisaInstrument:
    """
    A class to represent pyvisa objects

    Attributes
    ----------
    _instrument: pyvisa.resources.Resource
        The pyvisa resource
    _isConnected: bool
        True if the instrument has been connected, false otherwise
    _logger: logging.Logger
        A logger specified by python's logger module
    _resourceAddress: str
        The address of the instrument
    _resourceManager: pyvisa.highlevel.ResourceManager
        A handle to the resource manager provided by pyvisa
    _resourceName: str
        The name of the resource

    Methods
    -------
    connect(self)
        Opens the visa connection to the instrument.
    disconnect(self)
        Closes the visa connection to the instrument.
    write(self,query:str)
        Writes a string to the instrument.
    query_ascii(self, query:str)
        Sends a query to the instrument and reads the response.
    set_logger(self, logger: logging.Logger)
        Sets the logger that handles the message output for the _print function
    _print(self, msg:str, level:str)
        Prints a message with either python's 'print' function or the logger provided in 'set_logger'
    """
    _instrument: pyvisa.Resource = None
    _isConnected: bool = False
    _logger: logging.Logger = None
    _loggingLevels: dict = {'NOTSET': logging.NOTSET,
                            'DEBUG': logging.DEBUG,
                            'INFO': logging.INFO,
                            'WARNING': logging.WARNING,
                            'ERROR': logging.ERROR,
                            'CRITICAL': logging.CRITICAL}
    _resourceAddress: str = None
    _resourceManager: pyvisa.highlevel.ResourceManager = None

    def __init__(self, resource_address: str, resource_name: str, resource_manager: pyvisa.highlevel.ResourceManager):
        """
        Parameters
        ----------
        resource_address: str
            The address of the instrument
        resource_name: str
            The name of the resource
        resource_manager: pyvisa.highlevel.ResourceManager
            An instance of pyvisa's resource manager
        """
        self._resourceAddress = resource_address
        self._resourceName = resource_name
        if isinstance(resource_manager, pyvisa.ResourceManager):
            self._resourceManager = resource_manager
        else:
            msg = 'The second argument of the constructor should be ' + \
                  'an instance of pyvisa.ResourceManager'
            raise TypeError(msg)

    def connect(self):
        if not self._isConnected:
            self._instrument = self._resourceManager.open_resource(self._resourceAddress)
            self._isConnected = True
        else:
            msg = 'Instrument \'{0}\' already open on address \'{1}\'.'.format(self._resourceName,
                                                                               self._resourceAddress)
            self._print(msg=msg, level='WARNING')

    def disconnect(self):
        if (self._instrument is not None) and self._isConnected:
            self._instrument.close()
            self._isConnected = False
        else:
            msg = "Instrument '{0}' already closed.".format(self._resourceName)
            self._print(msg=msg, level='WARNING')

    def write(self, string: str):
        """
        Parameters
        ----------
        string: str
            The string to write to the instrument
        """
        if not self._isConnected:
            msg = 'Please connect the instrument before attempting to write to it. Duh!'
            raise errors.InstrumentError(self._resourceAddress, self._resourceName, msg)
        if isinstance(string, str):
            self._instrument.write(string)
        else:
            raise TypeError('Invalid query: \'{0}\'.'.format(string))

    def query_ascii(self, q: str, **kwargs):
        """
        Parameters
        ----------
        q: str
            The query to send to the insturment
        **kwargs:
            keyword arguments

        Returns
        -------
        str
            The response from the instrument
        """
        # converter = kwargs.get('converter', None)  # <- Implement later
        # separator = kwargs.get('separator', None)  # <- Implement later
        is_array = kwargs.get('is_array', False)  # <- Tries to convert to numpy array
        values = None
        if isinstance(q, str):
            if is_array:
                values = self._instrument.query_ascii_values(q, container=np.array)
            else:
                values = self._instrument.query_ascii_values(q)
        return values

    def set_logger(self, logger: logging.Logger):
        """
        Parameters
        ----------
        logger: logging.Logger
            The logger to handle the system messages

        Raises
        ------
        Warning
            If the logger is not an instance of logging.Logger
        """
        if isinstance(logger, logging.Logger):
            self._logger = logger
        else:
            msg: str = 'The logger should be an instance of \'logging.Logger\'.'
            raise Warning(msg)

    def _print(self, msg: str, level="INFO"):
        """

        Parameters
        ----------
        msg: str
            The message to print
        level: level
            The level of the message
        """
        level_no = self._loggingLevels[level]
        if self._logger is None:
            print(msg)
        elif isinstance(self._logger, logging.Logger):
            self._logger.log(level_no, msg)

    def __del__(self):
        self.disconnect()
