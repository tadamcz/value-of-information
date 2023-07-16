from __future__ import annotations  # This will become the default in Python 3.10.

import os
import statistics
import warnings

import numpy as np
import pandas as pd
import scipy.stats
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen
import float_table
from value_of_information import constants, voi
from value_of_information import utils
from value_of_information.rounding import round_sig
from tabulate import tabulate

class SimulationParameters:
	def __init__(self, prior, bar, study_sample_size=None, population_std_dev=None, sd_B=None):
		"""
		We can call T the parameter over which we want to conduct inference,
		and B the random variable we observe. Realisations of B can be denoted b.
		"""
		self.prior_T = prior
		self.prior_T_ev = self.prior_T.mean()
		if study_sample_size is None and population_std_dev is None:
			if sd_B is None:
				raise ValueError
		else:
			if study_sample_size is None or population_std_dev is None:
				raise ValueError
			sd_B = population_std_dev / np.sqrt(study_sample_size)

		self.sd_B = sd_B
		self.study_sample_size = study_sample_size
		self.population_std_dev = population_std_dev  # Assumed to be known
		self.bar = bar
		self.likelihood_is_normal = True  # Always the case for now
		self.discrete_choice = True  # Always the case for now

	def prior_family(self):
		if isinstance(self.prior_T, scipy.stats._distn_infrastructure.rv_frozen):
			prior_fam = self.prior_T.dist.__class__.__name__
		else:
			prior_fam = self.prior_T.__class__.__name__
		return prior_fam

	def __repr__(self):

		information = {
			"Prior family": self.prior_family(),
			"Bar": self.bar,
			"E[T]": round_sig(self.prior_T_ev, constants.ROUND_SIG_FIG),
			"sd(B)": round_sig(self.sd_B, constants.ROUND_SIG_FIG),
			"Study sample size": self.study_sample_size,
		}
		return pd.DataFrame([information]).to_string(index=False)


class SimulationExecutor:
	def __init__(self, input: SimulationParameters, force_explicit_bayes=False, force_explicit_b_draw=True,
				 print_every=None):
		self.input = input
		self.do_explicit_bayes = force_explicit_bayes or (not self.input.likelihood_is_normal)
		self.do_explicit_b_draw = force_explicit_b_draw or (not self.input.discrete_choice)
		self.print_every = print_every

	def execute(self, max_iterations=None, convergence_target=0.1, iterations=None) -> SimulationRun:
		"""
		If `iterations` is not `None`, there will be exactly that many iterations.

		Otherwise, the simulations stops
		after `standard_error_of_mean < convergence_target*mean` is reached,
		or after `max_iterations` iterations, whichever comes first.
		"""
		self.print_explainer()
		print("\n" + self.input.__repr__())
		print(f"\nExplicit Bayesian update: {self.do_explicit_bayes}")
		print(f"Explicit b_i draws: {self.do_explicit_b_draw}\n")
		if max_iterations is None:
			if self.do_explicit_bayes:
				max_iterations = 1000
			else:
				max_iterations = 100_000
		if iterations is None:
			max_iterations = max_iterations
			convergence_target = convergence_target
		else:
			max_iterations = iterations
			convergence_target = None

		this_run = SimulationRun(self.input, self)

		# For each iteartion i of the simulation, we draw a true value T_i from the prior.
		# For efficiency, it's better to do this outside the loop
		# See: https://github.com/scipy/scipy/issues/9394
		T_is = self.input.prior_T.rvs(size=max_iterations)

		if self.do_explicit_b_draw:
			# For each iteration i of the simulation, we draw a
			# distance (b_i-T_i), outside the loop for efficiency.
			# Note: this won't work for every likelihood function.
			b_i_distances = stats.norm(0, self.input.sd_B).rvs(size=max_iterations)

		if not self.do_explicit_bayes:
			if self.print_every is not None:
				print_intermediate_every = self.print_every
			elif max_iterations is not None:
				print_intermediate_every = max_iterations // 10
			else:
				print_intermediate_every = 1000
			threshold_b = voi.solve_threshold_b(self.input.prior_T, self.input.sd_B, self.input.bar)
		else:
			print_intermediate_every = self.print_every or 10
			threshold_b = None

		if convergence_target is None:
			print(f"The simulation will run for exactly {iterations} iterations.")
		else:
			utils.print_wrapped(
				f"The simulation will stop after `standard_error_of_mean < {convergence_target}*mean` is reached, "
				f"or after {max_iterations} iterations, whichever comes first.")
		i = 0
		while i < max_iterations:
			T_i = T_is[i]
			if self.do_explicit_b_draw:
				iteration = self.iteration_explicit_b_draw(T_i, i, b_i_distances, threshold_b)
			else:
				iteration = self.iteration_decision_distribution(T_i, threshold_b)

			iteration = self.add_indices_to_dict_keys(iteration)
			this_run.append(iteration)

			# Repeatedly calling `.sem()` is expensive
			if iterations is None and len(this_run) % 10 == 0:
				std_err = this_run.standard_error_mean_voi()
				mean = this_run.mean_voi()
				if std_err < convergence_target * mean:
					this_run.print_intermediate()
					print(f"Converged after {len(this_run)} simulation iterations!")
					break

			if len(this_run) % print_intermediate_every == 0:
				this_run.print_intermediate()

			i += 1
		else:
			print(
				f"Simulation ended after {len(this_run)} iterations. "
				f"Standard error of mean benefit from signal: {round_sig(this_run.standard_error_mean_voi())})")

		this_run.print_final()

		return this_run

	def iteration_explicit_b_draw(self, T_i, i, b_i_distances, threshold_b):
		"""
		We draw an estimate b_i from Normal(T_i,sd(B))
		"""

		# For efficiency, this is done outside the T_i-loop.
		# The lines below are equivalent to:
		# `b_i = stats.norm(T_i, sd_B_i).rvs(size=1)`
		b_i_distance = b_i_distances[i]
		b_i = T_i + b_i_distance

		return voi.value_of_information(T_i, self.input.sd_B, self.input.bar, self.input.prior_T, self.input.prior_T_ev,
										b=b_i,
										threshold_b=threshold_b, explicit_bayes=self.do_explicit_bayes)

	def iteration_decision_distribution(self, T_i, threshold_b):
		"""
		When dealing with a binary choice, for each `t` we can ask: what is the pr of each decision, i.e. what are the
		probabilities `P(d_1|T=t)` and `P(d_2|T=t)`?

		This is an an alternative to explicitly drawing `b_i`s.
		"""
		return voi.value_of_information(T_i, self.input.sd_B, self.input.bar, self.input.prior_T, self.input.prior_T_ev,
										threshold_b=threshold_b)

	def add_indices_to_dict_keys(self, iteration_dictionary):
		"""
		For display purposes
		"""

		mapping = {
			'T': 'T_i',
			'b': 'b_i',
			'E[T|b]': 'E[T|b_i]',
			'P(T>bar|b)': 'P(T>bar|b_i)',
			'E[T|b]>bar': 'E[T|b_i]>bar',
			'P(d_1|T)': 'P(d_1|T_i)',
			'P(d_2|T)': 'P(d_2|T_i)'
		}
		new_dictionary = {}
		for key in iteration_dictionary:  # iterate through them all because we want to keep the order of keys
			old_key = key
			try:
				new_key = mapping[old_key]
			except KeyError:
				new_key = old_key
			new_dictionary[new_key] = iteration_dictionary[old_key]

		return new_dictionary

	def print_explainer(self):
		utils.print_wrapped("We call T the parameter over which we want to conduct inference, "
							"and B the random variable (signal) we observe. Realisations of B are denoted b. "
							"Currently, only one distribution family is supported for B: the normal distribution with unknown mean T "
							"and known standard deviation.")


class SimulationRun:
	def __init__(self, inputs: SimulationParameters, executor: SimulationExecutor):
		self.input = inputs
		self.iterations_data = []
		self.do_explicit_bayes = executor.do_explicit_bayes
		self.do_explicit_b_draw = executor.do_explicit_b_draw

		if self.do_explicit_b_draw:
			self.voi_key = "VOI"
		else:
			self.voi_key = "E_B[VOI]"

	def __len__(self):
		return len(self.iterations_data)

	def append(self, iteration: dict):
		self.iterations_data.append(iteration)

	def mean_voi(self):
		return statistics.fmean([i[self.voi_key] for i in self.iterations_data])

	def standard_error_mean_voi(self):
		return scipy.stats.sem([i[self.voi_key] for i in self.iterations_data])

	def print_intermediate(self):
		if "PYTEST_CURRENT_TEST" in os.environ:
			return
		iteration_number = len(self.iterations_data)
		std_err = self.standard_error_mean_voi()
		mean = self.mean_voi()
		information = {
			"Iteration of simulation": iteration_number,
			"Mean VOI": round_sig(mean),
			"Standard error of mean": round_sig(std_err),
		}
		df = pd.DataFrame([information])
		print(df.to_string(index=False))

	def get_column(self, key):
		df = pd.DataFrame(self.iterations_data)
		return df[key]

	def print_final(self):
		if "PYTEST_CURRENT_TEST" in os.environ:
			return
		utils.print_wrapped(
			f"\nFor each iteration i of the simulation, we draw a true value T_i from the prior, and we draw "
			"an estimate b_i from Normal(T_i,sd(B)). The decision-maker cannot observe T_i, their subjective "
			"posterior expected value is E[T|b_i]. E[T|b_i] and P(T>bar|b_i) are only computed if "
			"running an explicit bayesian update. 'd_1' is the bar, and 'd_2' "
			"is the object of study.\n")
		# Once the display.max_rows is exceeded, the display.min_rows options determines how many rows are shown in
		# the truncated repr.
		with pd.option_context('display.max_columns', None, 'display.max_rows', 20,
							   'display.min_rows', 20,
							   'display.width', None):
			df = pd.DataFrame(self.iterations_data)
			print(float_table.format_df(df, sig_figs=3))

		mean_benefit_signal = self.mean_voi()
		sem_benefit_signal = self.standard_error_mean_voi()
		iterations = len(self.iterations_data)

		if mean_benefit_signal < 0:
			warnings.warn(
				f"VOI is negative with {iterations} simulation iterations. Try more iterations?")

		top_info = {
			"Mean VOI": mean_benefit_signal,
			"Standard error of mean VOI": sem_benefit_signal,
		}

		if self.do_explicit_bayes:
			top_info.update({
				"Mean of posterior expected values across iterations": self.get_column(
					'E[T|b_i]').mean(),
			})

		if self.do_explicit_b_draw:
			top_info.update({
				"Fraction of iterations where E[T|b_i] > bar":
					self.get_column("E[T|b_i]>bar").sum() / iterations,
			})

		df = pd.DataFrame([top_info]).T
		print(float_table.format_df(df, sig_figs=3).to_string(header=False))

		qs = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, .9, 0.95, .99, .999]
		voi_quantiles_info = {}
		# Quantiles of the VOI distribution
		for q in qs:
			if self.do_explicit_b_draw:
				voi_key = "VOI"
			else:
				voi_key = "E_B[VOI]"
			title = f"Quantiles of the {voi_key} distribution"
			voi_quantiles_info[q] = self.get_column(voi_key).quantile(q)

		df = pd.DataFrame([voi_quantiles_info]).T
		print("\n" + title)
		print(float_table.format_df(df, sig_figs=3).to_string(header=False))

		contributions_info = []
		# Contribution to the VOI of deciles of T_i
		for decile in range(0, 10):
			if self.do_explicit_b_draw:
				voi_key = "VOI"
			else:
				voi_key = "E_B[VOI]"
			title = f"Contribution to {voi_key} of deciles of T_i"

			dleft, dright = decile / 10, (decile + 1) / 10
			tleft, tright = self.get_column("T_i").quantile([dleft, dright])
			voi_sum = self.get_column(voi_key).sum()
			voi_contribution = self.get_column(voi_key)[
								   (self.get_column("T_i") >= tleft) & (
											   self.get_column("T_i") < tright)
								   ].sum() / voi_sum

			contributions_info.append({
				"Decile of T_i": f"{dleft} to {dright}",
				"From T_i": tleft,
				"To T_i": tright,
				"Contribution to VOI": voi_contribution,
			})

		df = pd.DataFrame(contributions_info)
		print("\n" + title)
		print("Note: these deciles are from the simulation and do not exactly match the theoretical deciles of T")
		df["Contribution to VOI"] = df["Contribution to VOI"].map("{:.0%}".format)
		df = float_table.format_df(df, sig_figs=3)
		print(tabulate(df, headers="keys", tablefmt="github", showindex=False))

		# Contribution to VOI of 1% bins [in 0-10% and 90-100%] of T_i
		contributions_info = []
		for bin in list(range(0, 10)) + list(range(90, 100)):
			if self.do_explicit_b_draw:
				voi_key = "VOI"
			else:
				voi_key = "E_B[VOI]"
			title = f"Contribution to {voi_key} of 1% bins of T_i"

			bleft, bright = bin / 100, (bin + 1) / 100
			tleft, tright = self.get_column("T_i").quantile([bleft, bright])
			voi_sum = self.get_column(voi_key).sum()
			voi_contribution = self.get_column(voi_key)[
								   (self.get_column("T_i") >= tleft) & (
											   self.get_column("T_i") < tright)
								   ].sum() / voi_sum

			contributions_info.append({
				"Bin of T_i": f"{bleft} to {bright}",
				"From T_i": tleft,
				"To T_i": tright,
				"Contribution to VOI": voi_contribution,
			})

		df = pd.DataFrame(contributions_info)
		print("\n" + title)
		print(f"Note: these bins are 1/10th as wide as the deciles above. "
			  f"Each bins contains {len(self.get_column(voi_key))//100} observations, and the "
			  f"contributions may be imprecisely estimated.")
		df["Contribution to VOI"] = df["Contribution to VOI"].map("{:.0%}".format)
		df = float_table.format_df(df, sig_figs=3)
		print("\nContributions in the bottom 10%")
		print(tabulate(df.iloc[:10], headers="keys", tablefmt="github", showindex=False))
		print("\nContributions in the top 10%")
		print(tabulate(df.iloc[10:], headers="keys", tablefmt="github", showindex=False))

	def csv(self):
		return pd.DataFrame(self.iterations_data).to_csv()

	@property
	def bar(self):
		return self.input.bar

	@property
	def prior(self):
		return self.input.prior_T

	@property
	def prior_family(self):
		return self.input.prior_family()
