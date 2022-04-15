import numpy as np
from scipy import stats
from metalogistic import MetaLogistic

from value_of_information.simulation import SimulationInputs, SimulationExecutor, SimulationRun
from value_of_information.study_cost_benefit import CostBenefitsExecutor, CostBenefitInputs

prior_mu, prior_sigma = 1, 1

inputs = SimulationInputs(
	prior=stats.lognorm(scale=np.exp(prior_mu), s=prior_sigma),
	sd_B=10,
	bar=6)

simulation_run = SimulationExecutor(inputs).execute()

cb_inputs = CostBenefitInputs(
	value_units="utils",
	money_units="M$",
	capital=100,
	study_cost=5,
	simulation_run=simulation_run
)

CostBenefitsExecutor(cb_inputs).execute()

