"""
Test 15: time step array of the stochastic reactors.

Builds the time step array (ParticlesCloud._set_time_step_array) for a constant time step
where time_max / dt is rounded down.
Check: the time marching loop of generate_stochastic_database (while time < time_max) does not
need more iterations than the array length, and a time step read as a string by YAML (1e-5)
raises a TypeError.

Independent of the other tests.
"""
from types import SimpleNamespace
import pytest
from ai_reacting_flows.stochastic_reactors_data_gen.particles_cloud import ParticlesCloud


def test_time_step_array_length():
    # 1.972e-3 / 1e-6 = 1971.9999999999998
    cloud = SimpleNamespace(dt_input=1.0e-6, time_max=1.972e-3)
    ParticlesCloud._set_time_step_array(cloud)

    # Same loop as in generate_stochastic_database
    time = 0.0
    iteration = 0
    while time < cloud.time_max:
        time += cloud.dt[iteration]
        iteration += 1

    assert iteration <= cloud.n_ite


def test_time_step_string():
    cloud = SimpleNamespace(dt_input="1e-5", time_max=1.0e-3)
    with pytest.raises(TypeError):
        ParticlesCloud._set_time_step_array(cloud)
