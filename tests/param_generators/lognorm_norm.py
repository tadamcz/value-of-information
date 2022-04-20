from typing import List

import numpy as np
from scipy import stats

from value_of_information.simulation import SimulationInputs

def linsp_mu(n) -> List[SimulationInputs]:
	inputs = []

	for prior_mu in np.linspace(0.5, 2, num=n):
		prior_sigma = 1
		prior = stats.lognorm(scale=np.exp(prior_mu), s=prior_sigma)

		i = SimulationInputs(
			prior=prior,
			sd_B=10,
			bar=5
		)
		inputs.append(i)
	return inputs
