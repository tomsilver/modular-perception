"""Example showing how to interact with a gym environment."""

import gymnasium as gym
import numpy as np

from modular_perception.query_types import SensorQuery
from modular_perception.utils import wrap_gym_env_with_sensor_module


def test_gym_api():
    """Example showing how to interact with a gym environment."""

    # Create the environment.
    env = gym.make("CartPole-v1")

    # Create a sensor module that captures observations whenever env.reset()
    # or env.step() are called.
    sensor_name = "gym_observation"
    env, sensor_module = wrap_gym_env_with_sensor_module(env, sensor_name)
    sensor_query = SensorQuery(sensor_name)

    obs, _ = env.reset()
    sensed_obs = sensor_module.get_response(sensor_query)
    assert np.allclose(obs, sensed_obs)
    for _ in range(3):
        sensor_module.tick()
        obs, _, _, _, _ = env.step(env.action_space.sample())
        sensed_obs = sensor_module.get_response(sensor_query)
        assert np.allclose(obs, sensed_obs)
