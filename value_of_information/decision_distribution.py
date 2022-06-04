from scipy import stats


def with_signal(prior_T, t, sd_B, bar, threshold=None, explicit_bayes=False):
	if explicit_bayes:
		return via_explicit_bayes(prior_T, sd_B, bar)
	else:
		if threshold is None:
			raise ValueError("Must provide `threshold` argument if `explicit_bayes=False`.")
		return via_threshold(threshold, t, sd_B)


def via_explicit_bayes(prior_T, sd_B, bar):
	# There should generally not be a reason to reach this path, but it could be included
	# in the future for completeness and to give another way to test_infinite_precision result consistency
	raise NotImplementedError


def via_threshold(threshold, t, sd_B):
	pr_choose_bar = stats.norm.cdf(threshold, loc=t, scale=sd_B)
	pr_choose_object = 1 - pr_choose_bar

	return {
		"pr_d_1": pr_choose_bar,
		"pr_d_2": pr_choose_object,
	}
