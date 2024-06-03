"""A module with Python functions that return sensor readings."""

from typing import Any, Callable, Dict, Hashable, TypeAlias

from modular_perception.perceiver import ModuleCannotAnswerQuery, PerceptionModule
from modular_perception.query_types import SensorQuery

SensorOutput: TypeAlias = Any


class SensorAgentModule(PerceptionModule[SensorQuery, SensorOutput]):
    """A module with Python functions that return sensor readings."""

    def __init__(
        self, sensors: Dict[str, Callable[[], SensorOutput]], *args, **kwargs
    ) -> None:
        self._sensors = sensors
        super().__init__(*args, **kwargs)

    def _get_response(self, query: Hashable) -> SensorOutput:
        if not isinstance(query, SensorQuery):
            raise ModuleCannotAnswerQuery
        try:
            sensor = self._sensors[query.name]
        except KeyError:
            raise ModuleCannotAnswerQuery
        return sensor()
