"""
Test 36: check of the transforms of the learning database (LearningDatabase.check_inputs).

With output_omegas, the outputs are f_Y(Y(t+dt)) - f_X(Y(t)), with f_X and f_Y the transforms of the inputs and
outputs (log_transform_X, log_transform_Y): an increment only if they are the same transform.
Check: log and Box-Cox mixed with output_omegas stop the processing; the same transform, outputs not transformed
(log_transform_Y: 0), or outputs that are not differences are accepted.

Independent of the other tests.
"""
import pytest

from ai_reacting_flows.databases_processing.database_processing import LearningDatabase


def database(log_transform_X, log_transform_Y, output_omegas):
    """LearningDatabase with only the parameters read by check_inputs (no database read)."""

    dtb = LearningDatabase.__new__(LearningDatabase)
    dtb.log_transform_X = log_transform_X
    dtb.log_transform_Y = log_transform_Y
    dtb.output_omegas = output_omegas
    dtb.rollout = False
    dtb.dt_var = False

    return dtb


@pytest.mark.parametrize("log_transform_X, log_transform_Y", [(1, 2), (2, 1)])
def test_mixed_transforms_with_omegas(log_transform_X, log_transform_Y):

    with pytest.raises(SystemExit, match="same transform"):
        database(log_transform_X, log_transform_Y, output_omegas=True).check_inputs()


@pytest.mark.parametrize("log_transform_X, log_transform_Y, output_omegas", [(1, 1, True), (2, 2, True), (1, 0, True),
                                                                             (2, 0, True), (1, 2, False), (2, 1, False)])
def test_accepted_transforms(log_transform_X, log_transform_Y, output_omegas):

    database(log_transform_X, log_transform_Y, output_omegas).check_inputs()
