import scipy.stats

from value_of_information.cost_benefit import CostBenefitsExecutor, CostBenefitParameters
from value_of_information.simulation import SimulationParameters, SimulationExecutor

prior = scipy.stats.logistic(10, 5)

params = SimulationParameters(
	prior=prior,
	sd_B=5,
	bar=12,
)

simulation_run = SimulationExecutor(params).execute()

cb_params = CostBenefitParameters(
	value_units="utils",
	money_units="M$",
	capital=100,
	signal_cost=5,
)

CostBenefitsExecutor(inputs=cb_params, simulation_run=simulation_run).execute()
