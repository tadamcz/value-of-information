from typing import List

from metalogistic import MetaLogistic

from value_of_information.simulation import SimulationParameters


def gen() -> List[SimulationParameters]:
	quantiles_array = [
		([0.1, 0.5, 0.9], [-20, -1, 50]),
		([0.1, 0.5, 0.9], [1, 5, 30]),
	]

	parameters = []
	for quantiles in quantiles_array:
		cdf_ps, cdf_xs = quantiles
		prior = MetaLogistic(cdf_ps=cdf_ps, cdf_xs=cdf_xs)
		parameters.append(SimulationParameters(
			prior=prior,
			sd_B=10,
			bar=5
		))

	return parameters
