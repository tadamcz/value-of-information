from functools import lru_cache

from bayes_continuous.likelihood_func import NormalLikelihood
from scipy import optimize
from sortedcontainers import SortedDict

from value_of_information import utils, bayes, decision_explicit_b, decision_distribution
from value_of_information.rounding import round_sig


@lru_cache(maxsize=100_000)
def solve_threshold_b(prior_T, sd_B, bar):
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
		likelihood = NormalLikelihood(b, sd_B)
		posterior = bayes.posterior(prior_T, likelihood)
		posterior_ev = posterior.expect()
		print(f"Trying b≈{round_sig(b, 5)}, which gives E[T|b]≈{round_sig(posterior_ev, 5)}")
		posterior_ev_sorted[b] = posterior_ev
		values_by_key = list(posterior_ev_sorted.values())
		if not utils.is_increasing(values_by_key, rtol=1e-9):
			raise RuntimeError(
				f"Found non-increasing sequence of E[T|b]: {values_by_key}. An integral was likely computed incorrectly.")
		return posterior_ev - bar

	p_0_1_T = prior_T.ppf(0.1)
	p_0_9_T = prior_T.ppf(0.9)

	left = p_0_1_T
	right = p_0_9_T

	# Setting the bracketing interval dynamically.
	FACTOR = 1.1

	additive_step = 1
	while f_to_solve(left) > 0.:
		additive_step = additive_step * FACTOR
		left = left - additive_step

	additive_step = 1
	while f_to_solve(right) < 0.:
		additive_step = additive_step * FACTOR
		right = right + additive_step
	# f_to_solve(left) and f_to_solve(right) now have opposite signs

	print(f"Running equation solver between b={round_sig(left, 5)} and b={round_sig(right, 5)}   ---->")
	x0, root_results = optimize.brentq(f_to_solve, a=left, b=right, full_output=True)
	print(f"Equation solver results for threshold value of b:\n{root_results}\n")

	return x0


def payoff(decision, T, bar):
	if decision == "d_1":
		return bar
	elif decision == "d_2":
		return T


def value_of_information(T, sd_B, bar, prior_T, prior_T_ev, b=None, threshold_b=None, explicit_bayes=False):
	"""
	prior_T_ev is passed in explicitly to avoid re-computing it each time
	"""
	explicit_b_draw = b is not None

	no_signal = decision_explicit_b.no_signal(prior_T_ev, bar)
	no_signal["payoff"] = payoff(no_signal["decision"], T, bar)

	if explicit_b_draw:
		with_signal = decision_explicit_b.with_signal(b, prior_T, sd_B, bar, explicit_bayes=explicit_bayes,
													  threshold=threshold_b)
		with_signal["payoff"] = payoff(with_signal["decision"], T, bar)
	else:
		with_signal = decision_distribution.with_signal(prior_T, T, sd_B, bar, threshold=threshold_b)

		payoff_d_1 = payoff("d_1", T, bar)
		payoff_d_2 = payoff("d_2", T, bar)

		with_signal["expected_payoff"] = payoff_d_1 * with_signal["pr_d_1"] + payoff_d_2 * with_signal["pr_d_2"]

	dictionary = {
		'T': T,
		'b': b,
		'w_out_signal': no_signal["decision"],
		'payoff_w_out_signal': no_signal["payoff"],
	}

	if explicit_bayes:
		dictionary.update({
			'E[T|b]': with_signal["posterior_ev"],
			'P(T>bar|b)': with_signal["pr_beat_bar"],
			'E[T|b]>bar': with_signal["posterior_ev"] > bar,
		})

	if explicit_b_draw:
		dictionary.update({
			'w_signal': with_signal["decision"],
			'payoff_w_signal': with_signal["payoff"],
			'E[T|b]>bar': with_signal["decision"] == "d_2",
			'VOI': with_signal["payoff"] - no_signal["payoff"],
		})
	else:
		dictionary.update({
			'P(d_1|T)': with_signal["pr_d_1"],
			'P(d_2|T)': with_signal["pr_d_2"],
			'E_B[payoff_w_signal]': with_signal["expected_payoff"],
			'E_B[VOI]': with_signal["expected_payoff"] - no_signal["payoff"],
		})

	return dictionary
