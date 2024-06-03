"""Utility functions."""

from typing import Any, Tuple

import gymnasium as gym

from modular_perception.modules.sensor_module import SensorAgentModule


class ObservationCaptureWrapper(gym.ObservationWrapper):
    """Helper for wrap_gym_env_with_sensor_module()."""

    def __init__(
        self,
        env: gym.Env,
    ) -> None:
        super().__init__(env)
        self._last_observation = None

    def observation(self, observation: Any) -> Any:
        self._last_observation = observation
        return observation

    def get_last_observation(self) -> Any:
        """Expose the last observation from reset() or step()."""
        return self._last_observation


def wrap_gym_env_with_sensor_module(
    env: gym.Env, sensor_name: str = "gym_observation"
) -> Tuple[gym.Env, SensorAgentModule]:
    """Create a wrapped verison of the environment, and a sensor module, so
    that whenever the original env resets or steps, the observations is piped
    to the sensory module."""
    env = ObservationCaptureWrapper(env)
    sensor_module = SensorAgentModule({sensor_name: env.get_last_observation})
    return env, sensor_module
