"""A module with functions for detecting objects of given types."""

from typing import Callable, FrozenSet, Hashable

from relational_structs import Object

from modular_perception.modules.sensor_module import SensorOutput
from modular_perception.perceiver import ModuleCannotAnswerQuery, PerceptionModule
from modular_perception.query_types import AllObjectDetectionQuery


class ObjectDetectionModule(PerceptionModule[AllObjectDetectionQuery, Object]):
    """A module with functions for detecting objects from string
    descriptions."""

    def __init__(
        self,
        object_detector: Callable[[SensorOutput], FrozenSet[Object]],
        sensory_input_query: Hashable,
        *args,
        **kwargs,
    ) -> None:
        self._object_detector = object_detector
        self._sensory_input_query = sensory_input_query
        super().__init__(*args, **kwargs)

    def _get_response(self, query: Hashable) -> Object:
        if not isinstance(query, AllObjectDetectionQuery):
            raise ModuleCannotAnswerQuery
        # Get the sensory input.
        sensory_input = self._send_query(self._sensory_input_query)
        # Run detection.
        return self._object_detector(sensory_input)
