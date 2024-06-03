"""Modules with functions for detecting ground atoms given predicates."""

from dataclasses import dataclass
from typing import Callable, Collection, Dict, FrozenSet, Hashable, Set, TypeAlias

import numpy as np
from numpy.typing import NDArray
from relational_structs import GroundAtom, Object, Predicate
from relational_structs.utils import get_object_combinations
from typing_extensions import Unpack

from modular_perception.perceiver import ModuleCannotAnswerQuery, PerceptionModule
from modular_perception.query_types import (
    ObjectFeatureQuery,
    PredicatesQuery,
)

FeatureDetector: TypeAlias = Callable[[Object, str], float]
PredicateInterpretation: TypeAlias = Callable[[FeatureDetector, Unpack[Object]], bool]


@dataclass(frozen=True)
class _LocalPredicatesQuery:
    """Necessary for dispatching."""

    predicates: FrozenSet[Predicate]
    objects: FrozenSet[Object]


@dataclass(frozen=True)
class _ImagePredicatesQuery:
    """Necessary for dispatching."""

    predicates: FrozenSet[Predicate]
    objects: FrozenSet[Object]


class LocalPredicateModule(PerceptionModule[PredicatesQuery, Set[GroundAtom]]):
    """Computes predicates based on object-centric features only."""

    def __init__(
        self,
        predicate_interpretations: Dict[Predicate, PredicateInterpretation],
        *args,
        **kwargs,
    ) -> None:
        self._predicate_interpretations = predicate_interpretations
        super().__init__(*args, **kwargs)

    def _get_response(self, query: Hashable) -> Set[GroundAtom]:
        if not isinstance(query, _LocalPredicatesQuery):
            raise ModuleCannotAnswerQuery
        predicates, objects = query.predicates, query.objects
        atoms: Set[GroundAtom] = set()
        detect_feature = lambda o, f: self._send_query(ObjectFeatureQuery(o, f))
        for pred in predicates:
            try:
                interp = self._predicate_interpretations[pred]
            except KeyError:
                raise ModuleCannotAnswerQuery
            for choice in get_object_combinations(objects, pred.types):
                if interp(detect_feature, *choice):
                    atoms.add(GroundAtom(pred, choice))
        return atoms


class ImagePredicateModule(PerceptionModule[PredicatesQuery, Set[GroundAtom]]):
    """Computes predicates based on images and object-centric features."""

    def __init__(
        self,
        detector: Callable[
            [
                Collection[Predicate],
                Collection[Object],
                FeatureDetector,
                Callable[[], NDArray[np.uint8]],
            ],
            Set[GroundAtom],
        ],
        image_query: Hashable,
        *args,
        **kwargs,
    ) -> None:
        self._detector = detector
        self._image_query = image_query
        super().__init__(*args, **kwargs)

    def _get_response(self, query: Hashable) -> Set[GroundAtom]:
        if not isinstance(query, _ImagePredicatesQuery):
            raise ModuleCannotAnswerQuery
        predicates, objects = query.predicates, query.objects
        detect_feature = lambda o, f: self._send_query(ObjectFeatureQuery(o, f))
        get_image = lambda: self._send_query(self._image_query)
        return self._detector(predicates, objects, detect_feature, get_image)


class PredicateDispatchModule(PerceptionModule[PredicatesQuery, Set[GroundAtom]]):
    """Separates predicates into the right types."""

    def __init__(
        self,
        local_predicates: Collection[Predicate],
        image_predicates: Collection[Predicate],
        *args,
        **kwargs,
    ) -> None:
        self._local_predicates = set(local_predicates)
        self._image_predicates = set(image_predicates)
        assert not (
            self._local_predicates & self._image_predicates
        ), "Predicates must be either local or image, not both"
        super().__init__(*args, **kwargs)

    def _get_response(self, query: Hashable) -> Set[GroundAtom]:
        if not isinstance(query, PredicatesQuery):
            raise ModuleCannotAnswerQuery
        predicates, objects = query.predicates, query.objects
        local_predicates: Set[Predicate] = set()
        image_predicates: Set[Predicate] = set()
        for predicate in predicates:
            if predicate in self._local_predicates:
                local_predicates.add(predicate)
            elif predicate in self._image_predicates:
                image_predicates.add(predicate)
            else:
                raise ModuleCannotAnswerQuery
        local_response = self._send_query(
            _LocalPredicatesQuery(frozenset(local_predicates), objects)
        )
        image_response = self._send_query(
            _ImagePredicatesQuery(frozenset(image_predicates), objects)
        )
        return local_response | image_response
