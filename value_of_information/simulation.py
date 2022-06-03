from __future__ import annotations  # This will become the default in Python 3.10.

import os
import statistics
import warnings

import numpy as np
import pandas as pd
import scipy.stats
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen

from value_of_information import constants, voi, decision_explicit_b, decision_distribution, decision
from value_of_information import utils
from value_of_information.rounding import round_sig


class SimulationInputs:
	def __init__(self, prior, bar, study_sample_size=None, population_std_dev=None, sd_B=None):
		"""
		We can call T the parameter over which we want to conduct inference,
		and B the random variable we observe. Realisations of B can be denoted b.
		"""
		self.prior_T = prior
		self.prior_ev = self.prior_T.expect()
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
			"E[T]": round_sig(self.prior_ev, constants.ROUND_SIG_FIG),
			"sd(B)": round_sig(self.sd_B, constants.ROUND_SIG_FIG),
			"Study sample size": self.study_sample_size,
		}
		return pd.DataFrame([information]).to_string(index=False)


class SimulationExecutor:
	def __init__(self, input: SimulationInputs, force_explicit_bayes=False, force_explicit_b_draw=True,
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

		# For each iteration_explicit_b i of the simulation, we draw a true value T_i from the prior.
		# For efficiency, it's better to do this outside the loop
		# See: https://github.com/scipy/scipy/issues/9394
		T_is = self.input.prior_T.rvs(size=max_iterations)

		if self.do_explicit_b_draw:
			# For each iteration_explicit_b i of the simulation, we draw a
			# distance (b_i-T_i), outside the loop for efficiency.
			# Note: this won't work for every likelihood function.
			b_i_distances = stats.norm(0, self.input.sd_B).rvs(size=max_iterations)

		if not self.do_explicit_bayes:
			print_intermediate_every = self.print_every or 1000
			threshold_b = voi.threshold_b(self.input.prior_T, self.input.sd_B, self.input.bar)
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

			# Our signal has the point estimator B_i for the parameter T_i.
			# sd(B_i) is a constant
			sd_B_i = self.input.sd_B

			if self.do_explicit_b_draw:
				iteration_output = self.iteration_explicit_b_draw(T_i, i, b_i_distances, sd_B_i, threshold_b)
			else:
				iteration_output = self.iteration_decision_distribution(T_i, threshold_b)

			this_run.append(iteration_output)

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

	def iteration_explicit_b_draw(self, T_i, i, b_i_distances, sd_B_i, threshold_b):
		"""
		We draw an estimate b_i from Normal(T_i,sd(B_i))
		"""

		# For efficiency, this is done outside the T_i-loop.
		# The lines below are equivalent to
		# `b_i = stats.norm(T_i, sd_B_i).rvs(size=1)`
		b_i_distance = b_i_distances[i]
		b_i = T_i + b_i_distance

		no_signal = decision_explicit_b.no_signal(self.input.prior_ev, self.input.bar)

		if self.do_explicit_bayes:
			with_signal = decision_explicit_b.with_signal(b_i, self.input.prior_T, self.input.sd_B, self.input.bar,
														  explicit_bayes=True)
		else:
			with_signal = decision_explicit_b.with_signal(b_i, self.input.prior_T, self.input.sd_B, self.input.bar,
														  threshold=threshold_b)
			for key in ["pr_beat_bar", "posterior_ev"]:
				no_signal[key] = None
				with_signal[key] = None

		no_signal["payoff"] = voi.payoff(no_signal["decision"], T_i, self.input.bar)
		with_signal["payoff"] = voi.payoff(with_signal["decision"], T_i, self.input.bar)

		return self.iteration_to_dict(no_signal, with_signal, T_i, b_i)

	def iteration_decision_distribution(self, T_i, threshold_b):
		"""
		When dealing with a binary choice, for each `t` we can ask: what is the pr of each decision, i.e. what are the
		probabilities `P(d_1|T=t)` and `P(d_2|T=t)`?

		This is an an alternative to explicitly drawing `b_i`s.
		"""
		distribution_w_signal = decision_distribution.with_signal(self.input.prior_T, T_i, self.input.sd_B,
																  self.input.bar, threshold_b,
																  explicit_bayes=self.do_explicit_bayes)

		decision_no_signal = decision.no_signal(self.input.prior_ev, self.input.bar)

		no_signal = decision_no_signal
		no_signal["payoff"] = voi.payoff(no_signal["decision"], T_i, self.input.bar)

		with_signal = distribution_w_signal
		payoff_d_1 = voi.payoff("d_1", T_i, self.input.bar)
		payoff_d_2 = voi.payoff("d_2", T_i, self.input.bar)
		with_signal["expected_payoff"] = payoff_d_1 * with_signal["pr_d_1"] + payoff_d_2 * with_signal["pr_d_2"]

		return self.iteration_to_dict(no_signal, with_signal, T_i, b_i=None)

	def iteration_to_dict(self, no_signal: dict, with_signal: dict, T_i, b_i):
		"""
		The final dictionary that will be used to store information about this iteration. At the end of the simulation,
		these dictionaries will become rows in a Pandas DataFrame.
		"""
		ret = {
			'T_i': T_i,
			'b_i': b_i,
			'w_out_signal': no_signal["decision"],
			'payoff_w_out_signal': no_signal["payoff"],
		}

		if self.do_explicit_bayes:
			ret.update({
				'E[T|b_i]': with_signal["posterior_ev"],
				'P(T>bar|b_i)': with_signal["pr_beat_bar"],
				'E[T|b_i]>bar': with_signal["posterior_ev"] > self.input.bar,
			})

		if self.do_explicit_b_draw:
			ret.update({
				'w_signal': with_signal["decision"],
				'E[T|b_i]>bar': with_signal["decision"] == "d_2",
				'payoff_w_signal': with_signal["payoff"],
				'VOI': with_signal["payoff"] - no_signal["payoff"],
			})
		else:
			ret.update({
				'P(d_1|T_i)': with_signal["pr_d_1"],
				'P(d_2|T_i)': with_signal["pr_d_2"],
				'E_B[payoff_w_signal]': with_signal["expected_payoff"],
				'E_B[VOI]': with_signal["expected_payoff"] - no_signal["payoff"],
			})

		return ret

	def print_explainer(self):
		utils.print_wrapped("We call T the parameter over which we want to conduct inference, "
							"and B the random variable (signal) we observe. Realisations of B are denoted b. "
							"Currently, only one distribution family is supported for B: the normal distribution with unknown mean T "
							"and known standard deviation.")


class SimulationRun:
	def __init__(self, inputs: SimulationInputs, executor: SimulationExecutor):
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
			f"\nFor each iteration_explicit_b i of the simulation, we draw a true value T_i from the prior, and we draw "
			"an estimate b_i from Normal(T_i,sd(B)). The decision-maker cannot observe T_i, their subjective "
			"posterior expected value is E[T|b_i]. E[T|b_i] and P(T|b_i > bar) are only computed if "
			"running an 'explicit' simulation. 'd_1' is the bar, and 'd_2' "
			"is the object of study.\n")
		# Once the display.max_rows is exceeded, the display.min_rows options determines how many rows are shown in
		# the truncated repr.
		with pd.option_context('display.max_columns', None, 'display.max_rows', 20, 'display.min_rows', 20,
							   'display.width', None, 'display.precision', 4):
			print(pd.DataFrame(self.iterations_data))

		mean_benefit_signal = self.mean_voi()
		sem_benefit_signal = self.standard_error_mean_voi()
		iterations = len(self.iterations_data)

		if mean_benefit_signal < 0:
			warnings.warn(
				f"VOI is negative with {iterations} simulation iterations. Try more iterations?")

		information = {
			"Mean VOI": mean_benefit_signal,
			"Standard error of mean VOI": sem_benefit_signal,
		}

		if self.do_explicit_bayes:
			information.update({
				"Mean of posterior expected values across iterations": self.get_column('E[T|b_i]').mean(),
			})

		if self.do_explicit_b_draw:
			information.update({
				"Fraction of iterations where E[T|b_i] > bar":
					self.get_column("E[T|b_i]>bar").sum() / iterations,
			})

		quantiles = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, .9, 0.95, .99, .999]
		for q in quantiles:
			if self.do_explicit_b_draw:
				key = f"Quantile {q} VOI"
				information[key] = self.get_column('VOI').quantile(q)
			else:
				key = f"Quantile {q} E_B[VOI]"
				information[key] = self.get_column('E_B[VOI]').quantile(q)

		df = pd.DataFrame([information]).T
		with pd.option_context('display.precision', 4):
			print("\n" + df.to_string(header=False))

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
