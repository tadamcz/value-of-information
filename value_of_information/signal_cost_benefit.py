from typing import Optional

import pandas as pd
from sigfig import round as round_sig

from value_of_information import utils
from value_of_information.simulation import SimulationRun


class CostBenefitInputs:
	def __init__(self, value_units, money_units, capital, signal_cost):
		"""
		:param value_units: For example: "utils", "multiples of GiveDirectly", or "lives saved"
		:param money_units: For example: "$" or "M$", or "£"
		:param capital: How much money do you have?
		:param signal_cost: Expressed in money_units
		"""
		self.signal_cost = signal_cost
		self.capital = capital
		self.money_units = money_units
		self.value_units = value_units


class CostBenefitsExecutor:  # todo add tests
	def __init__(self, inputs, simulation_run: Optional[SimulationRun] = None):
		"""
		:param simulation_run: An instance of class `SimulationRun`. It must be such that:
		- the prior and bar are expressed in value_units per money_units spent
		"""
		self.sim_run = simulation_run
		self.inputs = inputs

	def execute(self):
		if self.sim_run is None:
			raise ValueError
		prior_ev = self.sim_run.prior.expect()
		# Cost-effectiveness of money in the absence of the signal
		# Expressed in `prior_units`
		no_signal_best_option = max(self.sim_run.bar, prior_ev)

		prior_units = f"{self.inputs.value_units} per {self.inputs.money_units} spent"

		utils.print_wrapped(f"\nNote: you should make sure that the prior (a {self.sim_run.prior_family} with "
							f"mean {round_sig(prior_ev, 2)}) and the bar ({self.sim_run.bar}) are expressed in {prior_units}.")

		# Output:
		signal_benefit_per_usd_spent = self.sim_run.mean_voi()
		capital_after_signal = self.inputs.capital - self.inputs.signal_cost
		value_with_signal = capital_after_signal * (signal_benefit_per_usd_spent + no_signal_best_option)
		value_without_signal = self.inputs.capital * no_signal_best_option
		net_benefit_signal = value_with_signal - value_without_signal

		result = {
			f"Best option without signal ({prior_units})": no_signal_best_option,
			f"Capital ({self.inputs.money_units})": self.inputs.capital,
			f"Expected value without signal ({self.inputs.value_units})": value_without_signal,

			f"Expected benefit from signal ({prior_units})": signal_benefit_per_usd_spent,
			f"Capital left after signal ({self.inputs.money_units})": capital_after_signal,
			f"Expected value with signal ({self.inputs.value_units})": value_with_signal,

			f"Expected net benefit from signal ({self.inputs.value_units})": net_benefit_signal,
		}

		with pd.option_context('display.width', None, 'display.max_colwidth', None, 'display.precision', 4):
			df = pd.DataFrame([result]).T
			print("\n" + df.to_string(header=False))

		return result
