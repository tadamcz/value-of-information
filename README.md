# What this package does
This package lets you estimate the [value of information](https://en.wikipedia.org/wiki/Value_of_information) (VOI) of receiving a signal, in a simplified model.

It also lets you do a cost-benefit analysis: weighing the costs of the information against the stakes of the decision that may be improved by the information.

## Background
When we gain information about a decision-relevant quantity, that information may improve the decision we ultimately
make. The value of the (expected) improvement in the decision is
the value of information.

Sometimes, the information we gain tells us that one action is _certain_ to be better than another (for example, knowing
that tails came up in a coin toss implies we should bet on tails). But often the information is imperfect, and can only
pull our decision in the direction of optimality, in expectation.

Such imperfect information can be modelled as observing a random variable (or signal) `B`that is informative about the
true state of the world but contains some noise. The expected value of information is the expected benefit from
observing this random variable.

If the draw we observe is `b`, the true value of the quantity is `T` and `V` is the payoff function, the realised value
of information is:

```
V(decision_with_signal(b), T) - V(decision_without_signal, T) 
```

If a signal of `b` fails to change our decision, the value we realised is zero (regardless of `T`). This is intuitive.

When the signal does change our decision, the size of the benefit depends on the true state `T`, and on our decision
function `decision_with_signal`, which in turn depends on how the distribution of `B` is related to `T`.

We can find the expected value of information by taking an appropriate expectation over states of the world of the
expression above.

## Model details

In this package, we make some simplifying assumptions:

* We model the decision problem as a binary choice between an option whose value is known with certainty (the "bar"),
  and an uncertain option whose value is `T`.
* The decision-maker is taken to be risk-neutral (and the expected VOI is computed from a risk-neutral stance as well).
* The problem is one-dimensional, i.e. `T` and `B` follow one-dimensional distributions.
* Currently, only one distribution family is supported for `B`: `B` has a normal distribution with unknown mean `T` and
  known standard deviation.

The prior over `T` can be any one-dimensional SciPy continuous distribution.

This tool uses a simulation to approximate the expectation mentioned in the previous section. Specifically, for each
iteration `i` of the simulation:

1. We draw a true value `T_i` from the decision-maker's prior `P(T)`.
2. We draw an estimate `b_i` from `Normal(T_i,sd(B))`.
3. We can then calculate the decision that would be made with and without access to the signal:
    * _With the signal._ The decision-maker's subjective posterior expected value is `E[T|b_i]`. If `E[T|b_i]>bar`, the
      decision-maker chooses the uncertain option, otherwise they choose the certain option.
    * _Without the signal._ If `E[T]>bar`, the decision-maker chooses the uncertain option, otherwise they choose the
      certain option.
5. We calculate the decision-maker's payoffs with and without access to the signal. If choosing the uncertain option,
   they get a payoff of `T_i`; the payoff for the certain option is `bar`.

In this implementation, we take that expectation according to the decision maker's prior `P(T)` (this is because `T_i`s
are drawn from `P(T)` in step 1). In a subjective bayesian sense, this means that we compute the expected VOI by the
lights of the decision-maker; a frequentist interpretation might be that the decision situation is drawn from a larger
reference class in which `T` follows `P(T)`, and we are computing the average VOI in that class.

These concepts need not coincide in general. We could without difficulty model the decision-maker as acting according
to `P(T)`, but nonetheless compute the value of information by the lights of another actor who believes `Q(T)` (or the
VOI in a reference class following `Q(T)`).

## Computational approach for normal likelihood
We make use of the following fact to speed up the simulation: 

> When the signal `B` is normally distributed, with
mean `T`, then, for any prior distribution over `T`, `E[T|B=b]` is increasing in `b`.
 
This was shown by [Andrews et al. 1972](assets/andrews1972.pdf) (Lemma 1). It was generalised
by [Ma 1999](assets/ma1999.pdf) (Corollary 1.3) to any likelihood function arising from a `B` that (i) has `T` as a location
parameter, and (ii) is strongly unimodally distributed.

Therefore, when `B` is normally distributed with mean `T` (the only distribution currently supported), by default we run a numerical
equation solver to find the threshold value `B=b_t`, such that `E[T|b]>bar` if and only if `b>b_t`. This is hundreds of times faster than explicitly computing the posterior probability distribution `P(T|b_i)` in each iteration (which can be achieved by passing `force_explicit=True`).

## Cost-benefit analysis
The console output is relatively self-explanatory. You can also read [`signal_cost_benefit.py`](value_of_information/signal_cost_benefit.py), which is short and simple. A more detailed explanation may be added here later.

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

inputs = SimulationInputs(
	prior=stats.lognorm(scale=np.exp(prior_mu), s=prior_sigma),
	sd_B=10,
	bar=6)

simulation_run = SimulationExecutor(inputs).execute()

cb_inputs = CostBenefitInputs(
	value_units="utils",
	money_units="M$",
	capital=100,
	signal_cost=5,
)

CostBenefitsExecutor(inputs=cb_inputs, simulation_run=simulation_run).execute()
```

# Run tests

```shell
# At the root
poetry run pytest -n auto
```

# Extra-slow tests

Achieving high precision means some tests take very long to run (the standard error is proportional to 1/sqrt(n) which
declines very slowly). These tests are not run by default. They can be run (e.g. in the cloud) by
executing `pytest_with_extra_slow.sh`.