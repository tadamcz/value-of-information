# Setup

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

# Run tests

```shell
# At the root
poetry run pytest -n auto
```

# Extra-slow tests

Achieving high precision means some tests take very long to run (the standard error is proportional to 1/sqrt(n) which
declines very slowly). These tests are not run by default. They can be run (e.g. in the cloud) by
executing `pytest_with_extra_slow.sh`.

# Mathematics

When the likelihood function is normal (i.e., it arises from a normally distributed observation), we make use of the
following fact to speed up computation: the expected value of the posterior is increasing in the value of the
observation.

More explicitly:

The observable random variable B has a normal distribution with known variance and with unknown mean T. We receive an
observation of B=b. For any prior distribution over T,

![img](assets/equation.svg)

This was shown by [Andrews et al. 1972](assets/andrews1972.pdf) (Lemma 1). It was generalised
by [Ma 1999](assets/ma1999.pdf) (Corollary 1.3) to any likelihood function arising from a B that (i) has T as a location
parameter, and (ii) is strongly unimodally distributed.
