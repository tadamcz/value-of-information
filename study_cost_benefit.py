import numpy as np
import pandas as pd
from scipy import stats

from simulation import Simulation

# For example: "utils", "multiples of GiveDirectly", or "lives saved"
value_units = "utils"

# For example: "$" or "M$", or "£"
money_units = "M$"

# Study characteristics
study_sample_size = 1000
population_std_dev = 20
study_cost = 5

# How much money do you have?
capital = 100

# Prior
prior_units = f"{value_units} per {money_units} spent"
prior_mu, prior_sigma = 1, 1
prior = stats.lognorm(scale=np.exp(prior_mu), s=prior_sigma)

# Funding "bar", or value of a certain alternative
# Expressed in `prior_units`
bar = 5

# Cost-effectiveness of money in the absence of the study
# Expressed in `prior_units`
no_study_best_option = max(bar, prior.expect())

# Simulation
simulation_run = Simulation(
	prior=prior,
	study_sample_size=study_sample_size,
	population_std_dev=population_std_dev,
	bar=bar).run(max_iterations=100)

# Output:
ev_study_per_usd_spent = simulation_run.mean_value_study()
capital_after_study = capital - study_cost
ev_with_study = capital_after_study * (ev_study_per_usd_spent + no_study_best_option)
ev_without_study = capital * no_study_best_option
net_gain_study = ev_with_study - ev_without_study

result = {
	f"Best option without study ({prior_units})": no_study_best_option,
	f"Capital ({money_units})": capital,
	f"Expected value without study ({value_units})": ev_without_study,

	f"Expected study value ({prior_units})": ev_study_per_usd_spent,
	f"Capital left after study ({money_units})": capital_after_study,
	f"Expected value with study ({value_units})": ev_with_study,

	f"Expected net gain from study ({value_units})": net_gain_study,
}

with pd.option_context('display.width', None):
	print(pd.DataFrame([result]).T)
