# What this package does
This package lets you estimate the [value of information](https://en.wikipedia.org/wiki/Value_of_information) (VOI) of receiving a signal, in a simplified model.

It also lets you do a cost-benefit analysis: weighing the costs of the information against the stakes of the decision that may be improved by the information.

A simplified interface to this package is available at [valueofinfo.com](https://valueofinfo.com/).

# Tests
This package has a robust set of unit tests and end-to-end tests.

[`tadamcz/bayes-continuous`](https://github.com/tadamcz/bayes-continuous) is a dependency of this package used to compute
the Bayesian updates by numerical integration. It has its own set of tests, based on closed-form expressions that are available for [conjugate priors](https://en.wikipedia.org/wiki/Conjugate_prior).

## Background
When we gain information about a decision-relevant quantity, that information may improve the decision we ultimately
make. The value of the (expected) improvement in the decision is
the value of information.

Sometimes, the information we gain tells us that one action is _certain_ to be better than another (for example, knowing
that tails came up in a coin toss implies we should bet on tails). But often the information is imperfect, and can only
pull our decision in the direction of optimality, in expectation.

Such imperfect information can be modelled as observing a random variable (or signal) `B` that is informative about the
true state of the world `T` but contains some noise. The expected value of information is the expected benefit from
observing this random variable.

The realised value of information is:

```
VOI(T,B) = U(decision(B), T) - U(decision_0, T) 
```

where `U` is the payoff function, `decision` is the decision function when we have access to the signal, and `decision_0` is the decision we make in the absence of the signal.

If a signal of `B` fails to change our decision, the value we realised is zero (regardless of `T`). This is intuitive.

When the signal does change our decision, the size of the benefit depends on the true state `T`, and on our decision
function `decision_with_signal`, which in turn depends on how the distribution of `B` is related to `T`.

For each `T=t`, the expected value of information is

```
VOI(t) = E_B[VOI(T,B) | T=t] = E_B[U(decision(B), T) - U(decision_0, T) | T=t]
```

where `E_B` indicates that we're taking expectations with respect to (i.e. over the distribution of) `B`.

We can then find the entirely unconditional expected VOI `V` by taking expectations of the above with respect to `T`:

```
V = E_T[ E_B[VOI(t,b) | T=t]] 
```

Of course we might also, by the law of iterated expectations, write `V=E[VOI(t,b)]`, where the expectation sign without a subscript means the expectation is taken with respect to the joint distribution of `T` and `B`. The more explicit version is helpful for remembering that `V` is a double integral, which will make it easier to understand this model both conceptually and computationally.


## Model details

In this package, we make some simplifying assumptions:

* We model the decision problem as a **binary choice**.
    * This means `decision(B)` can take only two values `d_1` and `d_2` (and `decision_0` is equal to one of them).
    * This simplifies `VOI(t)` in the following way. For each `t`, instead of taking expectations of `VOI(t,B)` over infinitely many values of `B|T=t`, we can ask: what is the probability of each decision, i.e. what are the probabilities `P(d_1|T=t)` and `P(d_2|T=t)`?
* The binary choice is between:
    * **the bar** (`d_1`): an option with an expected payoff of `bar` about which we cannot gain additional information. Expressed mathematically, the inability to gain additional information means that `U(d_1, T)` is independent of `T`. So we can write `E[U(d_1)]=bar`. (It's
      irrelevant whether or not there is uncertainty over the payoff `U(d_1)`, what matters here is that this uncertainty is independent of `T` so we cannot gain additional information).
    * **the object of study** (`d_2`): an uncertain option whose payoff is `T`, about which we can gain additional information. The simplification here is that `U(d_2, T)=T`, but a more complicated dependence `U(d_2, T)=f(T)` could easily be modeled.
* The decision-maker is rational, i.e. upon receiving a signal of `B=b` they update their prior `P(T)` to `P(T|B=b)`. They risk-neutrally maximise expected `U`, which means they choose the object of study if and only if `E[T|B=b]>bar` (or `E[T]>bar` in the absence of the signal).
* The problem is one-dimensional, i.e. `T` and `B` follow one-dimensional distributions.
* Currently, only one distribution family is supported for `B`: `B` has a normal distribution with unknown mean `T` and
  known standard deviation.

The prior over `T` can be any one-dimensional SciPy continuous distribution.

To recapitulate, it may be helpful to think about how we might simulate this process. In each simulation iteration `i`:
1. We draw a true value `t_i` from the decision-maker's prior `P(T)`.
2. We draw an estimate `b_i` from `Normal(t_i,sd(B))`.
3. We can then calculate the decision that would be made with and without access to the signal:
    * _With the signal._ The decision-maker's subjective posterior expected value is `E[T|b_i]`. If `E[T|b_i]>bar`, the
      decision-maker chooses the object of study, otherwise they choose the bar.
    * _Without the signal._ If `E[T]>bar`, the decision-maker chooses the object of study, otherwise they choose the
      bar.
5. We calculate the decision-maker's payoffs with and without access to the signal. If choosing the object of study,
   they get a payoff of `T_i`; the payoff for the bar is `bar`.

Drawing `t_i` corresponds to the outer expectation `E_T[]` discussed above, and drawing `b_i` (dependent on `t_i`) corresponds to the inner expectation `E_B[]`. As we noted, for a discrete choice (in our case a binary one) the inner expectation does not require an integral, so explicitly drawing a `b_i` like in step 2 is not necessary, but it will give a correct estimate for `V`. In addition, explicitly drawing a `b_i` might be easier to think about and check the correctness of: each simulation iteration considers a fully specified world + observation pair `t_i,b_i` drawn from the joint distribution of `T` and `B`. 

Drawing `b_i` can be disabled by setting `force_explicit_b_draw=False`


### Remarks
Astute readers will have noticed another simplification. In calculating `V`, we take expectations over `T` according to the decision maker's prior `P(T)` (this is because `T_i`s
are drawn from `P(T)` in step 1). In a subjective bayesian sense, this means that we compute the expected VOI by the
lights of the decision-maker; a frequentist interpretation might be that the decision situation is drawn from a larger
reference class in which `T` follows `P(T)`, and we are computing the average VOI in that class.

These concepts need not coincide in general. We could without difficulty model the decision-maker as acting according
to `P(T)`, but nonetheless compute the value of information by the lights of another actor who believes `Q(T)` (or the
VOI in a reference class following `Q(T)`).

Analogously, `V` is calculated according to the same values as the decision-maker's values, i.e. it is modeled from a risk-neutral `U`-maximisation perspective, but this need not be so. (Technically this assumption is already present in the first section of this document).

## Computational shortcut: skipping the decision-maker's Bayesian update
For each value of `T=t` that we consider (in an integral or a simulation), we have to calculate `P(d_2|T=t)`, the probability that the object of study will be chosen. Or, if considering pairs `t,b` drawn from the joint distribution, we have to calculate `decision(b)` for each pair. Both of these would depend on the decision maker's subjective posterior distribution `P(T|B)`. Computing `P(T|B)` is computationally expensive in general. 

However, we can make use of the following fact to avoid explicitly computing the posterior each time: 

> When the signal `B` is normally distributed, with
mean `T`, then, for any prior distribution over `T`, `E[T|B=b]` is increasing in `b`.
 
This was shown by [Andrews et al. 1972](assets/andrews1972.pdf) (Lemma 1). It was generalised
by [Ma 1999](assets/ma1999.pdf) (Corollary 1.3) to any likelihood function arising from a `B` that (i) has `T` as a location
parameter, and (ii) is strongly unimodally distributed.

In these cases, instead of explicitly computing the posterior for every `b`-value, we
1. First run a numerical equation solver to find the threshold value `b_*` ("b-star"), such that `E[T|B=b]>bar` if and only if `b>b_*`.
2. Then, simply compare subsequent `b`-values to `b_*`.

This is hundreds of times faster than explicitly computing the posterior probability distribution `P(T|B=b)` for many `b`-values.

The shortcut and can be applied whether we are calculating `P(d_2|T=t)` (binary choice approach) or `decision(b)` (`t,b` pairs approach).

The shortcut can be disabled by passing `force_explicit_bayes=True`

Note: Currently `B~Normal(T,sd(B))` is the only distribution supported for `B`, so this computational shortcut is always applicable (i.e. "these cases" are all cases).

## Expression for `VOI(t)` in terms of `b_*`
We can now once again recapitulate and get the following expression:
```
VOI(t)
= E_B[VOI(t,B) | T=t]
= E_B[U(decision(b), t) - U(decision_0, t) | T=t]
= P(B>b_* | T=t)*U(d_2, t) + P(B<b_* | T=t)*U(d_1, t) - U(decision_0, t)
= P(B>b_* | T=t)*t + P(B<b_* | T=t)*bar - U(decision_0, t)
```

Given that `B` follows a normal distribution with mean `T` and CDF `F`:  
```
VOI(t,B)
= (1-F(b_*))*t + F(b_*)*bar - U(decision_0, t)
= F(b_*) * (bar-t) + t - U(decision_0, t)
```


## Closed-form expressions for `b_*`
If `P(T|B)` can be calculated in analytically, it follows that `b_*` can be calculated analytically. In the case of a normal prior over `T` with parameters `μ`, `σ` ([conjugate](https://en.wikipedia.org/wiki/Conjugate_prior) to the normal likelihood function for `B`)

```
E[T|B=b]>bar

iff

 b*sd(B)^2 + mu*σ^2      
──────────────────── > bar
   sd(B)^2 + σ^2 

iff

     sd(B)^2 * (bar - μ)    
b >  ─────────────────── + bar       
            σ^2        
```

It may be possible to use this result to directly calculate `V` in closed form as well, but I have not yet found such an expression.

Tests of [`tadamcz/bayes-continuous`](https://github.com/tadamcz/bayes-continuous) (a dependency of this package) use closed form solutions as well.

## Example: VOI for venture capital [TODO]
The previous section is quite abstract. It may be helpful to walk through a concrete example where our simplifications are reasonable, and the model is therefore a suitable one.

## Estimating expectations
These could be estimated either by Monte Carlo simulation, or by explicit numerical integration. The current version uses simulation.


## Cost-benefit analysis
The cost-benefit analysis assumes:
- "Choosing" the bar or the object of study means spending one's capital implementing that option. The amount of capital may vary.
- `T` and `bar` are expressed in terms of value realised _per unit of capital_. For example, "deaths averted per million dollars" or "new clients per dollar".
- The decision-maker can choose to spend `signal_cost` to acquire the signal. All other capital is spent implementing the option with the highest expected value.

This model is well-suited when choosing between different options that can absorb flexible amounts of capital (e.g. venture capital, ad spend, or philanthropy). However, it should be easy to model the costs and benefits differently, while leaving unchanged the functionality concerned with quantifying the value of information, which is more general and is the main contribution of this package. 

The console output should be relatively self-explanatory. The calculations can be read in [`signal_cost_benefit.py`](value_of_information/cost_benefit.py).

# Installation

Clone:

```shell
git clone https://github.com/tadamcz/value-of-information
cd value-of-information
```

Set up virtual environment:

```shell
poetry install
```

Run example

```shell
poetry run python example.py
```

# Usage
See `example.py`:
```python
prior_mu, prior_sigma = 1, 1

params = SimulationParameters(
	prior=stats.lognorm(scale=np.exp(prior_mu), s=prior_sigma),
	sd_B=10,
	bar=6)

simulation_run = SimulationExecutor(params).execute()

cb_params = CostBenefitParameters(
	value_units="utils",
	money_units="M$",
	capital=100,
	signal_cost=5,
)

CostBenefitsExecutor(inputs=cb_params, simulation_run=simulation_run).execute()
```

# Run tests

```shell
# At the root
poetry run pytest -n auto
```

# Origin of this project

This work was done under contract for [Open Philanthropy](https://www.openphilanthropy.org/). Open Philanthropy plans to use this tool as one input into the decision of whether to fund randomized trials in global health or development. Because the concept is quite general, we hope that the tool can also be useful to others.