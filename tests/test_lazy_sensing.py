"""Example showing that sensors are only called as-needed."""

import gymnasium as gym

from modular_perception.modules.sensor_module import SensorModule
from modular_perception.perceiver import (
    ModularPerceiver,
    ModuleCannotAnswerQuery,
    PerceptionModule,
)
from modular_perception.query_types import SensorQuery


def test_lazy_sensing():
    """Example showing that sensors are only called as-needed."""
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Create two "camera" sensors.
    num_camera1_calls = 0

    def _camera1():
        nonlocal num_camera1_calls
        num_camera1_calls += 1
        return env.render()

    num_camera2_calls = 0

    def _camera2():
        nonlocal num_camera2_calls
        num_camera2_calls += 1
        return env.render()

    sensors = {
        "camera1": _camera1,
        "camera2": _camera2,
    }

    sensor_module = SensorModule(sensors)

    # Create a dummy module that invokes the cameras every so often.
    class _CustomQuery:
        """Dummy query type."""

    class _CustomModule(PerceptionModule[_CustomQuery, None]):

        def _get_response(self, query):
            if not isinstance(query, _CustomQuery):
                raise ModuleCannotAnswerQuery
            # Query camera2 every 5 time steps. Never query camera1.
            if self._time % 5 == 0:
                img = self._send_query(SensorQuery("camera2"))
                del img  # not actually used

    custom_module = _CustomModule()

    # Finalize the perceiver.
    perceiver = ModularPerceiver({sensor_module, custom_module})

    seed = 0
    env.reset(seed=seed)
    perceiver.reset(seed)

    for _ in range(10):
        perceiver.tick()
        env.step(env.action_space.sample())
        sensed_obs = perceiver.get_response(_CustomQuery())
        assert sensed_obs is None

    assert num_camera1_calls == 0
    assert num_camera2_calls == 2
