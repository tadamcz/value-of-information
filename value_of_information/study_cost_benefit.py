from typing import Optional

import pandas as pd
from sigfig import round as round_sig

from value_of_information.simulation import SimulationRun


class CostBenefitInputs:
	def __init__(self, value_units, money_units, capital, study_cost):
		"""
		:param value_units: For example: "utils", "multiples of GiveDirectly", or "lives saved"
		:param money_units: For example: "$" or "M$", or "£"
		:param capital: How much money do you have?
		:param study_cost: Expressed in money_units
		"""
		self.study_cost = study_cost
		self.capital = capital
		self.money_units = money_units
		self.value_units = value_units


class CostBenefitsExecutor:  # todo add tests
	def __init__(self, inputs, simulation_run: Optional[SimulationRun]=None):
		"""
		:param simulation_run: An instance of class `SimulationRun`. It must be such that:
		- the prior and bar are expressed in value_units per money_units spent
		"""
		self.sim_run = simulation_run
		self.inputs = inputs

	def execute(self):
		print("\n")
		if self.sim_run is None:
			raise ValueError
		prior_ev = self.sim_run.prior.expect()
		# Cost-effectiveness of money in the absence of the study
		# Expressed in `prior_units`
		no_study_best_option = max(self.sim_run.bar, prior_ev)

		prior_units = f"{self.inputs.value_units} per {self.inputs.money_units} spent"

		print(f"Note: you should make sure that the prior (a {self.sim_run.prior_family} with "
			  f"mean {round_sig(prior_ev, 2)}) and the bar ({self.sim_run.bar}) are expressed in {prior_units}.")

		# Output:
		ev_study_per_usd_spent = self.sim_run.mean_value_study()
		capital_after_study = self.inputs.capital - self.inputs.study_cost
		ev_with_study = capital_after_study * (ev_study_per_usd_spent + no_study_best_option)
		ev_without_study = self.inputs.capital * no_study_best_option
		net_gain_study = ev_with_study - ev_without_study

		result = {
			f"Best option without study ({prior_units})": no_study_best_option,
			f"Capital ({self.inputs.money_units})": self.inputs.capital,
			f"Expected value without study ({self.inputs.value_units})": ev_without_study,

			f"Expected study value ({prior_units})": ev_study_per_usd_spent,
			f"Capital left after study ({self.inputs.money_units})": capital_after_study,
			f"Expected value with study ({self.inputs.value_units})": ev_with_study,

			f"Expected net gain from study ({self.inputs.value_units})": net_gain_study,
		}

		with pd.option_context('display.width', None):
			print(pd.DataFrame([result]).T)

		return result
