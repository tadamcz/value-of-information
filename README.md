# Setup

Clone:
```shell
git clone --recurse-submodules git@github.com:tadamcz/value-of-information.git
cd value-of-information
```

Set up virtual environment:
```shell
python3 -m venv .venv
source .venv/bin/activate # If on Windows, replace this line with with .venv\Scripts\activate
pip install -r requirements.txt
```

Run example
```shell
python example.py
```

# Development
* The use of a pip ["editable" install](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs) for the package `bayes_continuous` might confuse your IDE's index into thinking the files under that package don't exist. In PyCharm, I've found that `File -> Invalidate Caches` fixes the issue.

