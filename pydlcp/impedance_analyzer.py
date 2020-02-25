import os
import numpy as np
import pandas as pd
from pydlcp import visa_instrument as vi
import pyvisa
from io import StringIO
import time


# The return type for a voltage sweep
vcr_type = np.dtype([('V', 'd'), ('C', 'd'), ('R', 'd')])
# The return type for DLCP measurement
dlcp_type = np.dtype([('osc_level', 'd'),
                      ('bias', 'd'),
                      ('nominal_bias', 'd'),
                      ('V', 'd'),
                      ('C', 'd')])


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
        sweep_voltages = np.arange(voltage_start, voltage_stop+voltage_step, voltage_step)
        n_voltages = len(sweep_voltages)

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
        time.sleep(1)
        response = self.query("RUN")
        # values = values[:-1] # Remove character tail
        data = self.parse_impedance_data(response, n_voltages)
        time.sleep(1)
        # self.write('PSTOP')
        self.write('FNC2')
        self._status = "idle"
        return data

    def connect(self):
        super().connect()
        self._instrument.timeout = 60000
        # self._instrument.read_termination = '\n'
        # self._instrument.write_termination = '\n'

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
        program += "'20 SWM2',"  # Single Sweep
        program += "'30 IMP5',"  # Cs-Rs circuit
        program += "'40 SWP2',"  # DC Bias Sweep
        program += "'50 SWD1',"  # Sweep direction up
        program += "'60 {0}',"  # The integration time
        program += "'70 NOA={1}',"  # Number of Averages
        program += "'80 OSC={2:.4f};FREQ={3:.3E};BIAS={4:.4f}',"  # AC Amplitude (V), Frequency (Hz) & Bias (V)
        program += "'90 START={4:.3f};STOP={4:.3f}',"
        program += "'100 MANUAL={4:.4f}; NOP=2',"
        program += "'110 DTIME=100',"  # Delay time set to 100 ms
        program += "'120 SHT1',"  # Short Calibration set to On
        program += "'130 OPN1',"  # Open Calibration set to On
        program += "'140 AUTO',"  # Auto-Scale A & B
        program += "'150 CPYM2',"  # Copy Data Mode 2
        program += "'160 SWTRG',"  # Single Trigger Run
        program += "'170 COPY',"  # Copy Data to Instrument
        program += "'180 DCOFF',"  # Attempt to turn off DC Bias
        program += "'190 END'"
        return program

    def dlcp_sweep(self, nominal_bias: float, osc_start: float, osc_step: float, osc_stop: float,
                   frequency: float, **kwargs) -> np.ndarray:
        """
        Performs a DLCP sweep

        Parameters
        ----------
        nominal_bias: float
            The nominal bias for the DLCP sweep
        osc_start: float
            The starting amplitude for the oscillator level sweep (V)
        osc_step: float
            The step size for the amplitude of the oscillator level sweep (V)
        osc_stop: float
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

        ac_levels = np.arange(osc_start, osc_stop + osc_step, osc_step)
        n_levels = len(ac_levels)
        dc_bias = nominal_bias - 0.5*ac_levels
        results = np.empty(n_levels, dtype=dlcp_type)
        program_template = self._get_dlcp_template()
        self._status = "running"
        self.write('FNC2')
        for i, ac_level, bias in zip(range(n_levels), ac_levels, dc_bias):
            program = program_template.format(integration_time,
                                              number_of_averages,
                                              ac_level,
                                              frequency,
                                              bias)
            # self._print(program)
            self.write(program)
            time.sleep(1)
            response = self.query('RUN')
            time.sleep(1)
            # print(response)
            data = self.parse_dlcp_data(response, 20)
            self.write('FNC2')

            results[i] = (ac_level, bias, nominal_bias, data['V'][0], data['C'][0])
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

    def parse_dlcp_data(self, response: str, points: int):
        """
        Parses the response from the impedance analyzer for a DLCP measurement

        Parameters
        ----------
        response: str
            The response from the impedance analyzer
        points: int
            The number of points to average

        Returns
        -------
        vcr_type
            The parsed data
        """
        values = np.loadtxt(StringIO(response),
                            skiprows=1,
                            usecols=(1, 2, 3, 4),
                            dtype={'names': ('V', 'C', 'unit_factor_c', 'R'),
                                   'formats': ('d', 'd', 'U1', 'd')},
                            max_rows=points)

        factors_c = [self._multipliers[m] for m in values['unit_factor_c']]

        data = np.array([(np.mean(values['V']), np.mean(values['C'] * factors_c), np.mean(values['R']))],
                        dtype=vcr_type)

        return data

    def parse_impedance_data(self, response: str, max_rows: int, **kwargs) -> np.ndarray:
        """
        Parses the ASCII response from the Impedance Analyzer into an array of numbers
        Parameters
        ----------
        response: str
            The response from the Impedance Analyzer
        max_rows: int
            The number of values to read
        **kwargs:
            keyword arguments

        Returns
        -------
        np.ndarray
            The parsed response in the form of an array containing voltage, capacitance and resistance as columns
        """
        skip_rows = kwargs.get('skip_rows', 1)
        # self._print(response)
        # print('max_rows = {0:d}'.format(max_rows))
        values = np.loadtxt(StringIO(response),
                            skiprows=skip_rows,
                            usecols=(1, 2, 3, 4),
                            dtype={'names': ('V', 'C', 'unit_factor_c', 'R'),
                                   'formats': ('d', 'd', 'U1', 'd')},
                            max_rows=max_rows)
        # print(values)
        factors_c = [self._multipliers[m] for m in values['unit_factor_c']]
        data = np.empty(len(values), dtype=vcr_type)

        for i, v in enumerate(values):
            data[i] = (v['V'], v['C'] * factors_c[i], v['R'])
        # self._print(data)
        msg = 'Read impedance data from {0}'.format(self._resourceName)
        self._print(msg)
        return data

    def run_dlcp_one_pin(self, device_id: int, pin_num: int, start: float, stop: float, step: float):
        """
        Runs the dlcp measurement for a single pin

        Parameters
        ----------
        device_id: int
            The device id number that is being tested
        pin_num: int
            The pin number of device being tested
        start: float
            DC Bias start condition
        stop: float
            DC Bias stop condition
        step: float
            DC Bias step between each iteration
        """
        import os
        import platform
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib import cm

        cmap = cm.get_cmap('winter')
        fig, ax = plt.subplots()

        nominal_biases = np.arange(start=start, stop=(stop+step), step=step)
        normalize = Normalize(vmin=np.amin(nominal_biases), vmax=np.amax(nominal_biases))
        colors = [cmap(normalize(b)) for b in nominal_biases]
        for i, nb in enumerate(nominal_biases):
            data_dir = r"G:\Shared drives\FenningLab2\Projects\PVRD1\ExpData\DLCP\SiNx\D" + str(device_id) \
                       + "-p" + str(pin_num)
            if platform.system() == 'Windows':
                data_dir = r"\\?\\" + data_dir
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            filename = 'dlcp_D{0:d}_{1:.3}.csv'.format(device_id, nb)
            print('Runnig DLCP for nominal bias = {0:.3f}'.format(nb))
            dlcp_data = self.dlcp_sweep(nominal_bias=nb, start_amplitude=0.05, step_amplitude=0.05, stop_amplitude=1.0,
                                        frequency=1E6, noa=8, integration_time='ITM2')
            df = pd.DataFrame(data=dlcp_data)
            df.to_csv(os.path.join(data_dir, filename))
            ax.plot(dlcp_data['osc_level']*1000, dlcp_data['C']*1E12, 'o', color=colors[i])

        ax.set_xlabel('Oscillator Level (mV p-p)')
        ax.set_ylabel('C (pF)')
        plt.tight_layout(h_pad=0, w_pad=0)
        plt.show()
