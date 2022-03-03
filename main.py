from bayes_continuous.posterior import Posterior
from bayes_continuous.likelihood_func import NormalLikelihood
from scipy import stats

prior = stats.lognorm(1,1)
likelihood = NormalLikelihood(5,3)
print(Posterior(prior,likelihood).pdf(3))