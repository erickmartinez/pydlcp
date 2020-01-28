import numpy as np
from pydlcp import visa_instrument as vi
import pyvisa
from io import StringIO

# The return type for a voltage sweep
vcr_type = np.dtype([('V', 'd'), ('C', 'd'), ('R', 'd')])
# The return type for DLCP measurement
dlcp_type = np.dtype([('osc_level', 'd'),
                      ('bias', 'd'),
                      ('nominal_bias', 'd'),
                      ('V', 'd'),
                      ('C', 'd'),
                      ('R', 'd')])


class ImpedanceAnalyzer(vi.VisaInstrument):
    """
    A class used to represent HP4194A Impedance Analyzer and perform basic analysis for the bias-temperature stress
    (BTS) experiment.

    Attributes
    ----------
    _status: str
        The status of the instrument: either 'idle' or 'running'
    _waitTime: int
        Defines the time in seconds a measurement function will need to wait to retry execution if the instrument is
        busy with another measurement.

    Methods
    -------
    cv_sweep(self, voltage_start: float, voltage_step: float, voltage_stop: float, frequency: float, **kwargs)
        Tries to run a capacitance-voltage (CV) measurement sweep in the impedance analyzer according to the provided
        acquisition parameters. If the instrument is busy running another program, wait the specified amount of seconds
        defined in _waitTime

    _get_dlcp_template(self)
        Returns a template program for the DLCP measurement with placeholders for the relevant acquisition parameters.

    dlcp_sweep(self, nominal_bias: float, start_amplitude: float, step_amplitude: float, stop_amplitude: float,
                   frequency: float, **kwargs)
        Tries to run a drive-level capacitance profiling CV measurement program in the impedance analyzer. If the
        instrument is busy running another measurement, wait the specified amount of seconds defined in _waitTime.

    parse_impedance_data(self, response: str, **kwargs)
        Parses the ASCII response from the Impedance Analyzer into the python compound type vcr_type defined in this
        class
    """
    # The integration time.
    # ITM1: 500 us
    # ITM2: 5 ms
    # ITM3: 100 ms
    _integrationTime = ['ITM1', 'ITM2', 'ITM3']
    # Multipliers
    _multipliers = {'f': 1E-15, 'p': 1E-12, 'n': 1E-9, 'u': 1E-6, 'm': 1E-3}
    # Valid number of averages
    _noa = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # The sweep direction. SWD1 : sweep up, SWD2: sweep down
    _sweepDirection = ['SWD1', 'SWD2']
    # status of the instrument
    _status: str = 'idle'
    # Time to wait when the instrument is busy to take a new measurement (seconds)
    _waitTime: int = 60

    def __init__(self, address: str, resource_manager: pyvisa.ResourceManager, debug: bool = False):
        """
        Parameters
        ----------
        address: str
            The visa address of the resource
        resource_manager: pyvisa.ResourceManager
            The pyvisa resource manager used to get the resouce
        debug: bool
            True if in debug mode, false otherwise.
        """
        super().__init__(address, 'HP4194A', resource_manager)
        self._debug = debug

    def cv_sweep(self, voltage_start: float, voltage_step: float, voltage_stop: float, frequency: float,
                 **kwargs) -> np.ndarray:
        """
        Parameters
        ---------
        voltage_start: float
            The start voltage for the sweep (V)
        voltage_step: float
            The step of the voltage sweep (V)
        voltage_stop: float
            The stop voltage for the sweep (V)
        frequency: float
            The frequency of the AC signal (Hz)
        **kwargs:
            keyword arguments

        Returns
        -------
        np.ndarray:
            An array containing voltage, capacitance and resistance as columns

        Raises
        ------
        ValueError
            If the sweep direction is not valid (valid values are: SWD1 and SWD2).
        ValueError
            If the integration time is not valid (valid values are 'ITM1', 'ITM2' and 'ITM3')
        ValueError
            If the frequency < 0
        ValueError
            If the number of averages (noa) is invalid (valid values are 1, 2, 4, 8, 16, 32, 64, 128 and 256)
        """
        sweep_direction = kwargs.get('sweep_direction', 'SWD1')
        number_of_averages = kwargs.get('noa', 1)
        osc_amplitude = kwargs.get('osc_amplitude', 0.01)
        integration_time = kwargs.get('integration_time', 'ITM1')

        if sweep_direction not in self._sweepDirection:
            raise ValueError("Valid values for sweep direction are 'SWD1' and 'SWD2'")

        if integration_time not in self._integrationTime:
            raise ValueError("Valid values for integration time are '{0}'".format(self._integrationTime))

        if frequency <= 0:
            raise ValueError('The value of the frequency must be positive.')

        if number_of_averages not in self._noa:
            raise ValueError('Invalid number of averages.')

        # Check if the instrument is available
        program = "PROG"
        program += "'10 FNC1',"  # Impedance Measurement
        program += "'20 SWM2',"  # Single Mode Sweep
        program += "'30 IMP5',"  # Cs-Rs circuit
        program += "'40 SWP2',"  # DC Bias Sweep
        program += "'50 {0}',".format(sweep_direction)
        program += "'60 {0}',".format(integration_time)
        program += "'70 NOA={0}',".format(number_of_averages)  # Number of Averages
        program += "'80 OSC={0:.3f};FREQ={1:.3E}',".format(osc_amplitude, frequency)  # AC Amplitude (V) & Frequency
        program += "'90 START={0:.3f};STOP={1:.3f}',".format(voltage_start, voltage_stop)
        program += "'100 STEP={0:.4f}',".format(voltage_step)  # DC Sweep step magnitude
        program += "'110 DTIME=0',"  # Delay time set to 0
        program += "'120 SHT1',"  # Short Calibration set to On
        program += "'130 OPN1',"  # Open Calibration set to On
        program += "'140 AUTO',"  # Auto-Scale A & B
        program += "'150 CPYM2',"  # Copy Data Mode 2
        program += "'160 SWTRG',"  # Single Trigger Run
        program += "'170 COPY',"  # Copy Data to Instrument
        program += "'180 DCOFF',"  # Attempt to turn off DC Bias (Doesn't work)
        program += "'190 END'"

        self._status = "running"
        # Write the program on the impedance analyzer
        self.write(program)
        response = self.query('RUN')
        self.write('FNC2')
        # values = values[:-1] # Remove character tail
        data = self.parse_impedance_data(response)
        self._status = "idle"
        return data

    @staticmethod
    def _get_dlcp_template() -> str:
        """
        Returns
        -------
        str:
        Returns template string for the DLCP program
        """
        program = 'PROG'
        program += "'10 FNC1',"  # Impedance Measurement
        program += "'20 IMP5',"  # Cs-Rs circuit
        program += "'30 {0}',"  # The integration time
        program += "'40 NOA={1}',"  # Number of Averages
        program += "'50 OSC={2:.3f};FREQ={3:.3E};BIAS={4:.3E}',"  # AC Amplitude (V), Frequency (Hz) & Bias (V)
        program += "'60 DTIME=0',"  # Delay time set to 0
        program += "'70 SHT1',"  # Short Calibration set to On
        program += "'80 OPN1',"  # Open Calibration set to On
        program += "'90 AUTO',"  # Auto-Scale A & B
        program += "'100 CPYM2',"  # Copy Data Mode 2
        program += "'110 SWTRG',"  # Single Trigger Run
        program += "'120 COPY',"  # Copy Data to Instrument
        program += "'130 DCOFF',"  # Attempt to turn off DC Bias (Doesn't work)
        program += "'140 END'"
        return program

    def dlcp_sweep(self, nominal_bias: float, start_amplitude: float, step_amplitude: float, stop_amplitude: float,
                   frequency: float, **kwargs) -> np.ndarray:
        """
        Performs a DLCP sweep

        Parameters
        ----------
        nominal_bias: float
            The nominal bias for the DLCP sweep
        start_amplitude: float
            The starting amplitude for the oscillator level sweep (V)
        step_amplitude: float
            The step size for the amplitude of the oscillator level sweep (V)
        stop_amplitude: float
            The stop value for the oscillator level sweep (V)
        frequency: float
            The oscillator frequency (Hz)
        **kwargs:
            keyword arguments

        Returns
        -------
        np.ndarray
            The acquired capacitance data

        Raises
        ------
        ValueError
            If the integration time setting provided is not 'ITM1', 'ITM2' or 'ITM3'

        ValueError
            If the value provided for the frequency is less or equal to 0.

        ValueError
            If the number of averager (noa) is not valid. Valid values are:
            1, 2, 4, 8, 16, 32, 64, 128, 256
        """
        integration_time = kwargs.get('integration_time', 'ITM1')
        number_of_averages = kwargs.get('noa', 1)

        if integration_time not in self._integrationTime:
            raise ValueError("Valid values for integration time are '{0}'".format(self._integrationTime))

        if frequency <= 0:
            raise ValueError('The value of the frequency must be positive.')

        if number_of_averages not in self._noa:
            raise ValueError('Invalid number of averages.')

        ac_levels = np.arange(start_amplitude, stop_amplitude + step_amplitude, step_amplitude)
        n_levels = len(ac_levels)
        dc_bias = np.nonzero(nominal_bias + n_levels)
        results = np.empty(n_levels, dtype=dlcp_type)
        program_template = self._get_dlcp_template()
        self._status = "running"
        for i, ac_level, bias in zip(range(n_levels), ac_levels, dc_bias):
            program = program_template.format(integration_time,
                                              number_of_averages,
                                              ac_level,
                                              frequency,
                                              bias)
            self.write(program)
            response = self.query('RUN')
            self.write('FNC2')
            data = self.parse_impedance_data(response)
            results[i] = (ac_level, bias, nominal_bias, data['V'][0], data['C'][0], data['R'][0])
        self._status = "idle"
        return results

    @property
    def status(self) -> str:
        return self._status

    @property
    def wait_time(self) -> int:
        return self._waitTime

    @wait_time.setter
    def wait_time(self, seconds: int):
        if seconds > 0:
            self._waitTime = int(seconds)
        else:
            msg = 'The wait time must be a positive integer. Ignoring provided value of \'{0}\'.'.format(seconds)
            self._print(msg=msg, level='WARNING')

    def parse_impedance_data(self, response: str, **kwargs) -> np.ndarray:
        """
        Parses the ASCII response from the Impedance Analyzer into an array of numbers
        Parameters
        ----------
        response: str
            The response from the Impedance Analyzer
        **kwargs:
            keyword arguments

        Returns
        -------
        np.ndarray
            The parsed response in the form of an array containing voltage, capacitance and resistance as columns
        """
        skip_rows = kwargs.get('skip_rows', 2)
        values = np.loadtxt(StringIO(response),
                            skiprows=skip_rows,
                            dtype={'names': ('n', 'V', 'C', 'unit_factor_c', 'R', 'unit_factor_r'),
                                   'formats': ('i4', 'd', 'd', 'S1', 'd', 'S1')})
        factors_c = [self._multipliers[m] for m in values['unit_factor_c']]
        factors_r = [self._multipliers[m] for m in values['unit_factor_r']]
        data = np.empty(len(values), dtype=vcr_type)
        print(data)
        for i, v in enumerate(values):
            data[i] = (v['V'], v['C'] * factors_c, v['R'] * factors_r)
        msg = 'Read impedance data from {0}'.format(self._resourceName)
        self._print(msg)
        return data
