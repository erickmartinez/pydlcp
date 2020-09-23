import pydlcp.arduino_board as ard
import pydlcp.controller as controller
import configparser
import os


settings = r'G:\Shared drives\FenningLab2\LabData\ImpedanceAnalyzer\DLCP\20200922_training\D69_clean_low_frequency.ini'
arduino_com = 'COM8'
unit_name = 'HP1'
pin = 1

pinMappings = {
    'keithley': 'A0', 'fan': 'A1', 'thermocouple': '10', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: ' 7', 7: '8', 8: '9'
}

if __name__ == '__main__':
    if not os.path.exists(settings):
        raise FileExistsError('Settings file: \'{0}\' does not exist!'.format(settings))
    config = configparser.ConfigParser()
    config.read(settings)

    a = ard.ArduinoBoard(address=arduino_com, name=unit_name, pin_mappings=pinMappings)
    a.connect()
    dlcp_controller = controller.Controller()
    dlcp_controller.connect_devices()
    dlcp_controller.load_test_config(config=config)
    # a.connect_keithley()
    a.pin_on(2)
    try:
        dlcp_controller.start_dlcp()
    except Exception as e:
        print(e)
    finally:
        a.pin_off(2)
        dlcp_controller.disconnect_devices()
        a.disconnect()