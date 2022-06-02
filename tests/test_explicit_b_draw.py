from unittest.mock import patch

import numpy as np
import pytest

import tests.param_generators.lognorm_norm as gen_lognorm_norm
import tests.param_generators.norm_norm as gen_norm_norm
import tests.shared as shared
from value_of_information.simulation import SimulationExecutor


def do_nothing(*args, **kwargs):
	pass


class TestBDraw:
	def helper(self, inputs, iterations, relative_tolerance):
		with patch('value_of_information.simulation.SimulationRun.print_final') as patched_print:
			patched_print.side_effect = do_nothing

			b_draw_yes = SimulationExecutor(inputs, force_explicit_b_draw=True, print_every=1e9).execute(
				iterations=iterations)
			b_draw_no = SimulationExecutor(inputs, force_explicit_b_draw=False, print_every=1e9).execute(
				iterations=iterations // 10)

			assert b_draw_yes.mean_benefit_signal() == pytest.approx(
				b_draw_no.mean_benefit_signal(), rel=relative_tolerance)

			assert np.all(b_draw_no.get_column('b_i') == "NA")
			assert np.all(b_draw_yes.get_column('b_i') != "NA")

	@pytest.mark.parametrize('simulation_inputs',
							 argvalues=gen_lognorm_norm.linsp(8) + gen_norm_norm.linsp(12),
							 ids=shared.simulation_input_idfn)
	def test(self, simulation_inputs):
		self.helper(inputs=simulation_inputs, iterations=5_000_000, relative_tolerance=2 / 100)
