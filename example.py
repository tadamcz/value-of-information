import numpy as np
from scipy import stats

from value_of_information.cost_benefit import CostBenefitsExecutor, CostBenefitParameters
from value_of_information.simulation import SimulationParameters, SimulationExecutor

prior_mu, prior_sigma = 1, 1

params = SimulationParameters(
	prior=stats.lognorm(scale=np.exp(prior_mu), s=prior_sigma),
	sd_B=10,
	bar=6)

simulation_run = SimulationExecutor(params).execute()

cb_params = CostBenefitParameters(
	value_units="utils",
	money_units="M$",
	capital=100,
	signal_cost=5,
)

CostBenefitsExecutor(inputs=cb_params, simulation_run=simulation_run).execute()
