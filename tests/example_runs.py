import os
import runpy


def test_example_runs():
	path = os.path.join("..", "example.py")
	runpy.run_path(path)
