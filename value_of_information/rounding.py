import numpy as np
from sigfig import round as round_lib

from value_of_information import constants


def round_sig(x, sig=constants.ROUND_SIG_FIG, out_type=str, *args, **kwargs):
	if isinstance(x, (np.float64, np.int64)):
		if np.isnan(x):
			return "NaN"
		return round_lib(float(x), sig, type=out_type, *args, **kwargs)
	if isinstance(x, (float, int)):
		return round_lib(x, sig, type=out_type, *args, **kwargs)
	if isinstance(x, (list, np.ndarray)):
		return [round_sig(i, sig, out_type=out_type, *args, **kwargs) for i in x]
	else:
		raise TypeError(f"Must be numeric or list/array of numerics, not {type(x)}")
