"""A module with functions for detecting float object features."""

from typing import Any, Callable, Hashable, TypeAlias

from relational_structs import Object

from modular_perception.modules.sensor_module import SensorOutput
from modular_perception.perceiver import ModuleCannotAnswerQuery, PerceptionModule
from modular_perception.query_types import ObjectFeatureQuery

Feature: TypeAlias = Any


class ObjectFeatureModule(PerceptionModule[ObjectFeatureQuery, Feature]):
    """A module with functions for detecting object features."""

    def __init__(
        self,
        feature_detector: Callable[[SensorOutput, Object, str], Feature],
        sensory_input_query: Hashable,
        *args,
        **kwargs,
    ) -> None:
        self._feature_detector = feature_detector
        self._sensory_input_query = sensory_input_query
        super().__init__(*args, **kwargs)

    def _get_response(self, query: Hashable) -> Feature:
        if not isinstance(query, ObjectFeatureQuery):
            raise ModuleCannotAnswerQuery
        # Get the sensory input.
        sensory_input = self._send_query(self._sensory_input_query)
        # Run detection.
        return self._feature_detector(sensory_input, query.obj, query.feature)
