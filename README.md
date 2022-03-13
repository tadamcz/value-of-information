# Setup

Clone:
```shell
git clone --recurse-submodules https://github.com/tadamcz/value-of-information
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

# Development
* The use of a poetry editable install for the package `bayes_continuous` might confuse your IDE's index into thinking the files under that package don't exist. In PyCharm, I've found that `File -> Invalidate Caches` fixes the issue.

# Run tests
```shell
# At the root
poetry run pytest
```