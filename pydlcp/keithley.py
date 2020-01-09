
from pydlcp import visa_instrument as vi
import pyvisa


class Keithley(vi.VisaInstrument):
    """
    This class is used to represent a Keithley 2401 source meter in the context of a BTS measurement.

    Attributes
    ----------
    _debug: bool
        True if we instantiate it in debug mode.

    _sourceOff: bool
        True if the source is on, false otherwise

    Methods
    -------
    set_source_voltage(self, voltage: float)
        Sets the source voltage to the specified value.

    turn_source_on(self):
        Turns the keithley source on.

    turn_source_off(self):
        Turns the keithley source off.

    read(self):
        Sends a 'READ?' query to the keithley and returns the output.

    current(self):
        Reads the current from the keithley source-meter.
    """
    def __init__(self, address: str, resource_manager: pyvisa.ResourceManager, debug: bool = False):
        """
        Constructor for the class

        Parameters
        ----------
        address: str
            The visa address of the instrument
        resource_manager: pyvisa.ResourceManager
            The pyvisa reource manager used to get the resource.
        debug: bool
            True if in debug mode, false otherwise
        """
        super().__init__(address, 'Keihtley 2401', resource_manager)
        self._debug = debug
        self._sourceOn: bool = False

    def set_source_voltage(self, voltage: float):
        """
        Parameters
        ----------
        voltage: float
            The set point for the source voltage
        """
        q = ':SOUR:VOLT {0:.3E}'.format(voltage)
        self.write(q)

    def turn_source_on(self):
        self._sourceOn = True
        q = ':OUTP ON'
        self.write(q)

    def turn_source_off(self):
        self._sourceOn = False
        q = 'OUTP OFF'
        self.write(q)

    @property
    def source_on(self) -> bool:
        return self._sourceOn

    def read(self):
        q = ':READ?'
        self.query_ascii(q)

    @property
    def current(self):
        return self.read()

    def connect(self):
        super().connect()
        self.write('*RST')  # Reset K2401
        self.write(':OUTP:SMOD HIMP')  # Sets High Impedance Mode
        self.write(':ROUT:TERM REAR')  # Set I/O to Rear Connectors
        self.write(':SENS:FUNC:CONC OFF')  # Turn Off Concurrent Functions
        self.write(':SOUR:FUNC VOLT')  # Voltage Source Function
        self.write(":SENSE:FUNC 'CURR:DC'")  # DC Current Sense Function
        self.write(':SENSE:CURR:PROT .105')  # Set Compliance Current to 105 mA
        self.write(':SOUR:VOLT:MODE FIX')  # Set Voltage Source Mode to Fixed
        self.write(':SOUR:DEL .1')  # 100ms Source Delay (why?)
        self.write(':FORM:ELEM CURR')  # Select Data Collecting Item Current
        self.write(':SOUR:VOLT 0')  # Set bias voltage initially to 0

    def disconnect(self):
        self.turn_source_off()
        super().disconnect()
