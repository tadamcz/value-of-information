import numpy as np
import pytest

import tests.param_generators.lognorm_norm as gen_lognorm_norm
import tests.param_generators.norm_norm as gen_norm_norm
import tests.shared as shared
from value_of_information.simulation import SimulationExecutor


def helper(inputs, iterations, relative_tolerance):
	b_draw_yes = SimulationExecutor(inputs, force_explicit_b_draw=True).execute(
		iterations=iterations)
	b_draw_no = SimulationExecutor(inputs, force_explicit_b_draw=False).execute(
		iterations=iterations // 10)

	assert np.all(b_draw_no.get_column('b_i').to_numpy() == None)
	assert np.all(b_draw_yes.get_column('b_i').to_numpy() != None)

	assert b_draw_yes.mean_voi() == pytest.approx(b_draw_no.mean_voi(), rel=relative_tolerance)


@pytest.mark.parametrize('simulation_inputs',
						 argvalues=gen_lognorm_norm.linsp(4),
						 ids=shared.simulation_input_idfn)
def test_lognorm(simulation_inputs):
	helper(inputs=simulation_inputs, iterations=1_000_000, relative_tolerance=5 / 100)


@pytest.mark.parametrize('simulation_inputs',
						 argvalues=gen_norm_norm.linsp(6),
						 ids=shared.simulation_input_idfn)
def test_norm(simulation_inputs):
	helper(inputs=simulation_inputs, iterations=500_000, relative_tolerance=5 / 100)
