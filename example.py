from value_of_information.cost_benefit import CostBenefitsExecutor, CostBenefitParameters
from value_of_information.simulation import SimulationParameters, SimulationExecutor
from value_of_information.utils import lognormal

prior_mu, prior_sigma = 1, 1

prior = lognormal(prior_mu, prior_sigma)

params = SimulationParameters(
	prior=prior,
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
