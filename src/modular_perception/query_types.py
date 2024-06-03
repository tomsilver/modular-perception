"""Contains enums for discrete query types."""

from dataclasses import dataclass
from typing import FrozenSet

from relational_structs import Object, Predicate, Type


@dataclass(frozen=True)
class SensorQuery:
    """The identity of a sensor.

    This is a class so that it's easy to identify when a query is for a
    sensor.
    """

    name: str


@dataclass(frozen=True)
class AllObjectDetectionQuery:
    """A query to detect all seen objects."""

    object_types: FrozenSet[Type]


@dataclass(frozen=True)
class ObjectFeatureQuery:
    """A query to get a given feature of a given object."""

    obj: Object
    feature: str


@dataclass(frozen=True)
class PredicatesQuery:
    """A query to get all ground atoms for given predicates."""

    predicates: FrozenSet[Predicate]
    objects: FrozenSet[Object]


@dataclass(frozen=True)
class AllGroundAtomsQuery:
    """A query to get all ground atoms for all known predicates and objects."""
