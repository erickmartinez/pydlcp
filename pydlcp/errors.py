"""
This code provides custom exceptions for the pydlcp system

@author: Erick Martinez Loran <erickrmartinez@gmail.com>
"""


class Error(Exception):
    """
    Base class for exceptions in this module
    """
    pass


class InstrumentError(Error):
    """
    Exception raised when trying to access a pyvisa instrument that is not connected

    Attributes
    ----------
    _resourceAddress: str
        The address of the resource
    _resourceName: str
        The name of the resource (instrument)
    _message: str
        An explanation of the error
    """
    def __init__(self, address: str, resource_name: str, message: str):
        """
        Arguments
        ---------
        :param address: The address of the resource
        :type address: str
        :param resource_name: The name of the resource
        :type resource_name: str
        :param message: The explanation of the error
        :type message: str
        """
        self._resourceAddress = address
        self._resourceName = resource_name
        self._message = message

    @property
    def address(self) -> str:
        """
        :return: The address of the resource
        :rtype: str
        """
        return self._resourceAddress

    @property
    def resource_name(self) -> str:
        """
        :return: The name of the visa resource
        :rtype: str
        """
        return self._resourceName

    @property
    def message(self) -> str:
        """
        :return: The explanation of the error
        :rtype: str
        """
        return self._message


class HotplateError(InstrumentError):
    _heatingStatus = 1  # Not heating
    _temperatureSetpoint = 25

    def __init__(self, address: str, resource_name: str, message: str):
        super().__init__(address, resource_name, message)

    @property
    def heating_status(self) -> bool:
        return self._heatingStatus

    def set_heating_status(self, status: bool):
        self._heatingStatus = status

    @property
    def temperature_setpoint(self) -> int:
        return self._temperatureSetpoint

    def set_temperature_setpoint(self, setpoint: int):
        self._temperatureSetpoint = setpoint


class ConfigurationError(Error):
    """
    Base class for Configuration Errors

    Attributes
    ----------
    _message: str
        The explanation of the error
    """
    def __init__(self, message: str):
        """
        Arguments
        ---------
        :param message: The explanation of the error.
        :type message: str
        """
        self._message = message

    @property
    def message(self):
        return self._message


class SystemConfigError(ConfigurationError):
    def __init__(self, message: str, test_units: int):
        super().__init__(message=message)
        self._testUnits = test_units

    @property
    def test_units(self) -> int:
        return self._testUnits


class ArduinoError(InstrumentError):
    def __init__(self, address: str, name: str, message: str):
        super().__init__(address=address, resource_name=name, message=message)


class ArduinoSketchError(ArduinoError):
    def __init__(self, address: str, name: str, sketch_file: str, message: str):
        super().__init__(address=address, name=name, message=message)
        self._sketchFile = sketch_file

    @property
    def sketch_file(self) -> str:
        return self._sketchFile


