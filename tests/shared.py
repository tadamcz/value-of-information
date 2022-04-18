import bayes_continuous.utils
from bayes_continuous.likelihood_func import NormalLikelihood
from scipy import stats

from value_of_information.rounding import round_sig
from value_of_information.simulation import SimulationInputs


def get_location_scale(distribution: stats._distn_infrastructure.rv_frozen):
	args, kwds = distribution.args, distribution.kwds
	shapes, loc, scale = distribution.dist._parse_args(*args, **kwds)
	return loc, scale


def normal_normal_closed_form(normal_prior, normal_likelihood):
	if not bayes_continuous.utils.is_frozen_normal(normal_prior):
		raise ValueError
	if not isinstance(normal_likelihood, NormalLikelihood):
		raise ValueError

	mu_1, sigma_1 = get_location_scale(normal_prior)
	mu_2, sigma_2 = normal_likelihood.mu, normal_likelihood.sigma
	posterior_mu, posterior_sigma = bayes_continuous.utils.normal_normal_closed_form(mu_1, sigma_1, mu_2, sigma_2)
	posterior = stats.norm(posterior_mu, posterior_sigma)

	# For speed and accuracy, we can replace the method `expect` (numerical integration)
	# with `mean` (closed form).
	# Ideally, one would do this with patching/mocking instead of here.
	posterior.expect = posterior.mean

	return posterior


def simulation_input_idfn(inputs: SimulationInputs):
	pri_loc, pri_scale = get_location_scale(inputs.prior_T)
	return f"bar={inputs.bar}, E[T]~={round_sig(inputs.prior_ev)}, T_loc={pri_loc}, T_scale={pri_scale}, sd(B)~={inputs.sd_B}"


def rel_idfn(p):
	return f"rel={p}"


def iter_idfn(p):
	return f"iter={p}"


