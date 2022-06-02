from bayes_continuous.likelihood_func import NormalLikelihood

from value_of_information import bayes


def with_signal(b, prior_T, sd_B, bar, explicit_bayes=False, threshold=None):
	if explicit_bayes:
		return via_explicit_bayes(b, prior_T, sd_B, bar)
	else:
		if threshold is None:
			raise ValueError("Must provide `threshold` argument if `explicit_bayes=False`.")
		return via_threshold(b, threshold)


def no_signal(prior_ev, bar):
	if prior_ev > bar:
		return {"decision": "d_2"}
	else:
		return {"decision": "d_1"}


def via_explicit_bayes(b, prior_T, sd_B, bar):
	likelihood = NormalLikelihood(b, sd_B)
	posterior = bayes.posterior(prior_T, likelihood)
	pr_beat_bar = 1 - posterior.cdf(bar)
	posterior_ev = posterior.expect()

	if posterior_ev > bar:
		decision = "d_2"
	else:
		decision = "d_1"

	return {
		"decision": decision,
		"pr_beat_bar": pr_beat_bar,
		"posterior_ev": posterior_ev,
	}


def via_threshold(b, threshold):
	"""
	When the likelihood function is normal (i.e., it arises from a normally distributed observation), we make use
	of the following fact to speed up computation: the expected value of the posterior is increasing in the value
	of the observation. See README.md for more detail.

	So we can call this method, which only checks if the b passes the threshold that makes the posterior expected value
	greater than `bar`.
	"""
	if b > threshold:
		return {"decision": "d_2"}
	else:
		return {"decision": "d_1"}
