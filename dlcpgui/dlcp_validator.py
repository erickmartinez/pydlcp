from PyQt5.QtGui import QValidator
import re

_float_re = re.compile(r'(([+-]?\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)')
_invalid_intermediate_re = re.compile(r'([eE][eE]+)+')

class FloatValidator(QValidator):
	def __init__(self, min_val:float, max_val:float):
		super().__init__()
		self._min = min
		self._max = max

    def validate(self, string, position):
        if valid_float_string(string):
            return (QValidator.Acceptable, string, position)
        if valid_intermediate_str(string, position):
            return (QValidator.Intermediate, string, position)
        return (QValidator.Invalid, string, position)

    def fixup(self, text):
        match = _float_re.search(text)
        return match.groups()[0] if match else ""
		
def valid_float_string(string):
	match = _float_re.search(string)
	return match.groups()[0] == string if match else False

def valid_intermediate_str(string, position):
	match = _invalid_intermediate_re.search(string)
	first_check = False
	if string == "" or string[position-1] in 'e.-+' or string[position-1] in 'E.-+':
		first_check = True
	repeated_e = match.groups()[0] == string if match else False
	
	if repeated_e or first_check == False:
		return False
	
	return True
	
def validate_range(string, min_val, max_val):
	if max_val < float(string) or float(string) < min_val:
		return False
	return True
	