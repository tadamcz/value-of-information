from __future__ import annotations  # This will become the default in Python 3.10.

import warnings

import numpy as np
import pandas as pd
import scipy.stats
from bayes_continuous.likelihood_func import NormalLikelihood, LikelihoodFunction
from bayes_continuous.posterior import Posterior
from scipy import stats, optimize
from scipy.stats._distn_infrastructure import rv_frozen
from sortedcontainers import SortedDict
import statistics
import value_of_information.utils as utils

import value_of_information.constants as constants
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
	def __init__(self, input: SimulationInputs, force_explicit=False, print_every=None):
		self.input = input
		self.do_explicit = force_explicit or (not self.input.likelihood_is_normal)
		self.print_every = print_every

	def execute(self, max_iterations=None, convergence_target=0.1, iterations=None) -> SimulationRun:
		"""
		If `iterations` is not `None`, there will be exactly that many iterations.

		Otherwise, the simulations stops
		after `standard_error_of_mean < convergence_target*mean` is reached,
		or after `max_iterations` iterations, whichever comes first.
		"""
		self.print_explainer()
		print(self.input)
		print("\n")
		print(f"Explicit simulation: {self.do_explicit}")
		if max_iterations is None:
			if self.do_explicit:
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

		# For each iteration i of the simulation, we draw a true value T_i from the prior.
		# For efficiency, it's better to do this outside the loop
		# See: https://github.com/scipy/scipy/issues/9394
		T_is = self.input.prior_T.rvs(size=max_iterations)

		# For each iteration i of the simulation, we draw a
		# distance (b_i-T_i), outside the loop for efficiency.
		# Note: this won't work for every likelihood function.
		b_i_distances = stats.norm(0, self.input.sd_B).rvs(size=max_iterations)

		if not self.do_explicit:
			print_intermediate_every = self.print_every or 1000
			threshold_b = self.solve_for_threshold_b()
		else:
			print_intermediate_every = self.print_every or 10

		if convergence_target is None:
			print(f"The simulation will run for exactly {iterations} iterations.")
		else:
			print(f"The simulation will stop after `standard_error_of_mean < {convergence_target}*mean` is reached, "
				  f"or after {max_iterations} iterations, whichever comes first.")
		i = 0
		while i < max_iterations:
			T_i = T_is[i]

			# Our study has the point estimator B_i for the parameter T_i.
			# sd(B_i) is a constant
			sd_B_i = self.input.sd_B

			# We draw an estimate b_i from Normal(T_i,sd(B_i))
			# But for efficiency, this is done outside the loop.
			# The lines below are equivalent to
			# `b_i = stats.norm(T_i, sd_B_i).rvs(size=1)`
			b_i_distance = b_i_distances[i]
			b_i = T_i + b_i_distance

			iteration_kwargs = {
				'b_i':b_i,
				'T_i':T_i,
				'sd_B_i': sd_B_i,
			}
			if self.do_explicit:
				iteration_output = self.iteration_explicit_update(**iteration_kwargs)
			else:
				iteration_output = self.iteration_threshold_update(threshold_b=threshold_b, **iteration_kwargs)

			this_run.append(iteration_output)

			# Repeatedly calling `.sem()` is expensive
			if iterations is None and len(this_run) % 10 == 0:
				std_err = this_run.standard_error_mean_value_study()
				mean = this_run.mean_value_study()
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
				f"Standard error of mean study value: {round_sig(this_run.standard_error_mean_value_study())})")

		this_run.print_final()

		return this_run

	def iteration_explicit_update(self, b_i, T_i, sd_B_i):
		likelihood = NormalLikelihood(b_i, sd_B_i)

		posterior = self.posterior(self.input.prior_T, likelihood)

		return self.iteration_create_dict(posterior_explicit=posterior, T_i=T_i, b_i=b_i)

	def iteration_threshold_update(self, threshold_b, b_i, T_i, sd_B_i) -> dict:
		"""
		When the likelihood function is normal (i.e., it arises from a normally distributed observation), we make use
		of the following fact to speed up computation: the expected value of the posterior is increasing in the value
		of the observation. See README.md for more detail.

		So we can call this method, which only checks if the posterior expected value passes the threshold.
		"""

		if b_i > threshold_b:
			return self.iteration_create_dict(posterior_ev_beats_bar=True, T_i=T_i, b_i=b_i)
		else:
			return self.iteration_create_dict(posterior_ev_beats_bar=False, T_i=T_i, b_i=b_i)


	def iteration_create_dict(self, T_i, b_i, posterior_explicit=None, posterior_ev_beats_bar=None) -> dict:
		if posterior_explicit is None and posterior_ev_beats_bar is None:
			raise ValueError

		if posterior_explicit:
			posterior_ev = posterior_explicit.expect()
			pr_beat_bar = 1 - posterior_explicit.cdf(self.input.bar)
			posterior_ev_beats_bar = posterior_ev > self.input.bar
		else:
			pr_beat_bar = "not computed"
			posterior_ev = "not computed"

		# Without study
		if self.input.prior_ev > self.input.bar:
			decision_w_out_study = "candidate"
			value_w_out_study = T_i
		else:
			decision_w_out_study = "fallback"
			value_w_out_study = self.input.bar

		# With study
		if posterior_ev_beats_bar:
			decision_w_study = "candidate"
			value_w_study = T_i
		else:
			decision_w_study = "fallback"
			value_w_study = self.input.bar

		value_of_study = value_w_study - value_w_out_study

		iteration_output = {
			'T_i': T_i,
			'b_i': b_i,

			'E[T|b_i]': posterior_ev,
			'P(T|b_i>bar)': pr_beat_bar,

			'w_study': decision_w_study,
			'w_out_study': decision_w_out_study,

			'value_w_study': value_w_study,
			'value_w_out_study': value_w_out_study,

			'value_of_study': value_of_study,
		}

		return iteration_output

	def solve_for_threshold_b(self):
		"""
		We want to solve the following for b:
		```
		posterior_ev(b, ...) = bar
		```

		posterior_ev(b, ...) is increasing in b, so it has only one zero, and we can use
		Brent (1973)'s method as implemented in `scipy.optimize.brentq`.

		posterior_ev(b, ...) being increasing in b also has the consequence that we can
		set the bracketing interval for Brent's method dynamically.
		"""

		posterior_ev_sorted = SortedDict()  # Sorted by key
		def f_to_solve(b):
			likelihood = NormalLikelihood(b, self.input.sd_B)
			posterior = self.posterior(self.input.prior_T, likelihood)
			posterior_ev = posterior.expect()
			print(f"Trying b≈{round_sig(b, 5)}, which gives E[T|b]≈{round_sig(posterior_ev, 5)}")
			posterior_ev_sorted[b] = posterior_ev
			values_by_key = list(posterior_ev_sorted.values())
			if not utils.is_increasing(values_by_key, rtol=1e-9):
				raise RuntimeError(f"Found non-increasing sequence of E[T|b]: {values_by_key}. An integral was likely computed incorrectly.")
			return posterior_ev-self.input.bar

		p_0_1_T = self.input.prior_T.ppf(0.1)
		p_0_9_T = self.input.prior_T.ppf(0.9)

		left = p_0_1_T
		right = p_0_9_T

		# Setting the bracketing interval dynamically.
		FACTOR = 2

		additive_step = 1
		while f_to_solve(left) > 0.:
			additive_step = additive_step*FACTOR
			left = left - additive_step

		additive_step = 1
		while f_to_solve(right) < 0.:
			additive_step = additive_step*FACTOR
			right = right + additive_step
		# f_to_solve(left) and f_to_solve(right) now have opposite signs

		print(f"Running equation solver between b={round_sig(left, 5)} and b={round_sig(right, 5)}   ---->")
		x0, root_results = optimize.brentq(f_to_solve, a=left, b=right, full_output=True)
		print(f"Equation solver results for threshold value of b:\n{root_results}\n")

		return x0


	def posterior(self, prior: rv_frozen, likelihood: LikelihoodFunction):
		posterior = Posterior(prior, likelihood)
		if np.isnan(posterior.expect()):
			raise ValueError(f"Posterior expected value is NaN for {prior}, {likelihood}")
		return posterior

	def print_explainer(self):
		print("We call T the parameter over which we want to conduct inference, "
			  "and B the random variable we observe. Realisations of B are denoted b. "
			  "Currently, only normally distributed B is supported, where T is the mean "
			  "of B.\n")



class SimulationRun:
	def __init__(self, inputs: SimulationInputs, executor: SimulationExecutor):
		self.input = inputs
		self.iterations_data = []
		self.do_explicit = executor.do_explicit
		
	def __len__(self):
		return len(self.iterations_data)

	def append(self, iteration: dict):
		self.iterations_data.append(iteration)

	def mean_value_study(self):
		return statistics.fmean([i['value_of_study'] for i in self.iterations_data])

	def standard_error_mean_value_study(self):
		return scipy.stats.sem([i['value_of_study'] for i in self.iterations_data])

	def print_intermediate(self):
		iteration_number = len(self.iterations_data)
		std_err = self.standard_error_mean_value_study()
		mean = self.mean_value_study()
		information = {
			'Iteration of simulation': iteration_number,
			"Mean study value": round_sig(mean),
			"Standard error of mean": round_sig(std_err),
		}
		df = pd.DataFrame([information])
		print(df)

	def get_column(self, key):
		df = pd.DataFrame(self.iterations_data)
		return df[key]

	def print_final(self):

		print(f"\nFor each iteration i of the simulation, we draw a true value T_i from the prior, and we draw "
			  "an estimate b_i from Normal(T_i,sd(B)). The decision-maker cannot observe T_i, their subjective "
			  "posterior expected value is E[T|b_i]. E[T|b_i] and P(T|b_i > bar) are only computed if "
			  "running an 'explicit' simulation. 'fallback' is the option whose value is `bar`, and 'candidate' "
			  "is the uncertain option.")
		# Once the display.max_rows is exceeded, the display.min_rows options determines how many rows are shown in
		# the truncated repr.
		with pd.option_context('display.max_columns', None, 'display.max_rows', 20, 'display.min_rows', 20,
							   'display.width', None):
			print(pd.DataFrame(self.iterations_data))

		mean_value_of_study = self.mean_value_study()
		sem_of_study = self.standard_error_mean_value_study()
		iterations = len(self.iterations_data)

		if mean_value_of_study < 0:
			warnings.warn(f"Value of study is negative with {iterations} simulation iterations. Try more iterations?")

		information = {
			"Mean value of study": mean_value_of_study,
			"Standard error of mean value of study": sem_of_study,
		}

		if self.do_explicit:
			information.update({
				"Mean of posterior expected values across draws": self.get_column('E[T|b_i]').mean(),
				"Fraction of posterior means > bar":
					(self.get_column('E[T|b_i]') > self.input.bar).sum() / iterations,
			})

		quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
		for q in quantiles:
			key = f"Quantile {q} value of study"
			information[key] = self.get_column('value_of_study').quantile(q)

		df = pd.DataFrame([information]).T
		print(df)

	@property
	def bar(self):
		return self.input.bar

	@property
	def prior(self):
		return self.input.prior_T

	@property
	def prior_family(self):
		return self.input.prior_family()