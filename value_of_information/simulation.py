from __future__ import annotations  # This will become the default in Python 3.10.

import warnings

import numpy as np
import pandas as pd
import scipy.stats
from bayes_continuous.likelihood_func import NormalLikelihood, LikelihoodFunction
from bayes_continuous.posterior import Posterior
from scipy import stats, optimize
from scipy.stats._distn_infrastructure import rv_frozen

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
			"Prior EV": round_sig(self.prior_ev, constants.ROUND_SIG_FIG),
			"Study sample size": self.study_sample_size,
			"sd(B)": round_sig(self.sd_B, constants.ROUND_SIG_FIG),
		}
		return pd.DataFrame([information]).to_string(index=False)

class SimulationExecutor:
	def __init__(self, input: SimulationInputs, force_explicit=False):
		self.input = input
		self.do_explicit = force_explicit or (not self.input.likelihood_is_normal)

	def execute(self, max_iterations=None, convergence_target=0.1, iterations=None) -> SimulationRun:
		"""
		If `iterations` is not `None`, there will be exactly that many iterations.

		Otherwise, the simulations stops
		after `standard_error_of_mean < convergence_target*mean` is reached,
		or after `max_iterations` iterations, whichever comes first.
		"""
		print(self.input)
		print(f"Explicit simulation: {self.do_explicit}")
		if max_iterations is None:
			if self.do_explicit:
				max_iterations = 1000
			else:
				max_iterations = 100_000
		if iterations is None:
			max_iterations = max_iterations  # no need for self. here (or with self.this run)
			convergence_target = convergence_target
		else:
			max_iterations = iterations
			convergence_target = 0  # Can never be reached

		this_run = SimulationRun(self.input, self)

		# For each iteration i of the simulation, we draw a true value T_i from the prior.
		# For efficiency, it's better to do this outside the loop
		# See: https://github.com/scipy/scipy/issues/9394
		T_is = self.input.prior_T.rvs(size=max_iterations)

		if not self.do_explicit:
			print_intermediate_every = 1000
			threshold_b = self.solve_for_threshold_b()
		else:
			print_intermediate_every = 10

		i = 0
		while i < max_iterations:
			T_i = T_is[i]

			# Our study has the point estimator B_i for the parameter T_i.
			# sd(B_i) is a constant
			sd_B_i = self.input.sd_B

			# We draw an estimate b_i from Normal(T_i,sd(B_i)).
			b_i = stats.norm(T_i, sd_B_i).rvs()

			iteration_kwargs = {
				'b_i':b_i,
				'T_i':T_i,
				'sd_B_i': sd_B_i,
			}
			if self.do_explicit:
				iteration_output = self.iteration_explicit_update(**iteration_kwargs)
			else:
				iteration_output = self.iteration_threshold_update(threshold_b=threshold_b, **iteration_kwargs)

			iteration_output = pd.DataFrame([iteration_output])
			if this_run.iterations_data is None:
				this_run.iterations_data = iteration_output
			else:
				this_run.iterations_data = pd.concat([this_run.iterations_data, iteration_output], ignore_index=True)

			std_err = this_run.iterations_data['value_of_study'].sem()
			mean = this_run.iterations_data['value_of_study'].mean()
			if std_err < convergence_target * mean:
				this_run.print_intermediate()
				print(f"Converged after {len(this_run.iterations_data)} simulation iterations!")
				break
			if len(this_run.iterations_data) % print_intermediate_every == 0:
				this_run.print_intermediate()

			i += 1
		else:
			print(
				f"Did not converge after {len(this_run.iterations_data)} simulation iterations. "
				f"Standard error of mean study value: {round_sig(this_run.iterations_data['value_of_study'].sem())})")

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

			'posterior_ev': posterior_ev,
			'pr_beat_bar': pr_beat_bar,

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

		def f_to_solve(b):
			likelihood = NormalLikelihood(b, self.input.sd_B)
			posterior = self.posterior(self.input.prior_T, likelihood)
			posterior_ev = posterior.expect()
			print(f"Trying b≈{round_sig(b)}, which gives E[T|b]≈{round_sig(posterior_ev)}")
			return posterior_ev-self.input.bar


		p_0_1_T = self.input.prior_T.ppf(0.1)
		p_0_9_T = self.input.prior_T.ppf(0.9)

		left = p_0_1_T
		right = p_0_9_T

		# Setting the bracketing interval dynamically.
		# The approach is loosely inspired by scipy's `ppf`, see below
		# https://github.com/scipy/scipy/blob/b5d8bab88af61d61de09641243848df63380a67f/scipy/stats/_distn_infrastructure.py#L1826-L1844
		additive_step = 2
		FACTOR = 2
		while f_to_solve(left) > 0.:
			additive_step = additive_step*FACTOR
			left = left - additive_step

		additive_step = 2
		while f_to_solve(right) < 0.:
			additive_step = additive_step*FACTOR
			right = right + additive_step
		# f_to_solve(left) and f_to_solve(right) now have opposite signs

		print(f"Running equation solver between b={round_sig(left)} and b={round_sig(right)}   ---->")
		x0, root_results = optimize.brentq(f_to_solve, a=left, b=right, full_output=True)
		print(f"Equation solver results for threshold value of b:\n{root_results}\n")

		return x0


	def posterior(self, prior: rv_frozen, likelihood: LikelihoodFunction):
		posterior = Posterior(prior, likelihood)
		if np.isnan(posterior.expect()):
			raise ValueError(f"Posterior expected value is NaN for {prior}, {likelihood}")
		return posterior


class SimulationRun:
	def __init__(self, inputs: SimulationInputs, executor: SimulationExecutor):
		self.input = inputs
		self.iterations_data = None
		self.do_explicit = executor.do_explicit

	def print_intermediate(self):
		if self.iterations_data is None:
			raise ValueError
		iteration_number = len(self.iterations_data)
		std_err = self.iterations_data['value_of_study'].sem()
		mean = self.iterations_data['value_of_study'].mean()
		information = {
			'Iteration of simulation': iteration_number,
			"Mean study value": round_sig(mean),
			"Standard error of mean": round_sig(std_err),
		}
		df = pd.DataFrame([information])
		print(df)

	def print_final(self):
		if self.iterations_data is None:
			raise ValueError

		# Once the display.max_rows is exceeded, the display.min_rows options determines how many rows are shown in
		# the truncated repr.
		with pd.option_context('display.max_columns', None, 'display.max_rows', 20, 'display.min_rows', 20,
							   'display.width', None):
			print(self.iterations_data)

		mean_value_of_study = self.iterations_data['value_of_study'].mean()
		sem_of_study = self.iterations_data['value_of_study'].sem()
		iterations = len(self.iterations_data)

		if mean_value_of_study < 0:
			warnings.warn(f"Value of study is negative with {iterations} simulation iterations. Try more iterations?")

		information = {
			"Mean value of study": mean_value_of_study,
			"Standard error of mean value of study": sem_of_study,
		}

		if self.do_explicit:
			information.update({
				"Mean of posterior expected values across draws": self.iterations_data['posterior_ev'].mean(),
				"Fraction of posterior means > bar":
					(self.iterations_data['posterior_ev'] > self.input.bar).sum() / iterations,
			})

		quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
		for q in quantiles:
			key = f"Quantile {q} value of study"
			information[key] = self.iterations_data['value_of_study'].quantile(q)

		df = pd.DataFrame([information]).T
		print(df)

	def mean_value_study(self):
		return self.iterations_data['value_of_study'].mean()

	@property
	def bar(self):
		return self.input.bar

	@property
	def prior(self):
		return self.input.prior_T

	@property
	def prior_family(self):
		return self.input.prior_family()