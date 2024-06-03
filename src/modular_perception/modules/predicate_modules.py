"""Modules with functions for detecting ground atoms given predicates."""

from typing import Callable, Collection, Dict, Hashable, Set, TypeAlias

import numpy as np
from numpy.typing import NDArray
from relational_structs import GroundAtom, Object, Predicate
from relational_structs.utils import get_object_combinations
from typing_extensions import Unpack

from modular_perception.perceiver import ModuleCannotAnswerQuery, PerceptionModule
from modular_perception.query_types import (
    ImagePredicateQuery,
    LocalPredicateQuery,
    ObjectFeatureQuery,
)

FeatureDetector: TypeAlias = Callable[[Object, str], float]
PredicateInterpretation: TypeAlias = Callable[[FeatureDetector, Unpack[Object]], bool]


class LocalPredicateAgentModule(PerceptionModule[LocalPredicateQuery, Set[GroundAtom]]):
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
        if not isinstance(query, LocalPredicateQuery):
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


class ImagePredicateAgentModule(PerceptionModule[ImagePredicateQuery, Set[GroundAtom]]):
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
        if not isinstance(query, ImagePredicateQuery):
            raise ModuleCannotAnswerQuery
        predicates, objects = query.predicates, query.objects
        detect_feature = lambda o, f: self._send_query(ObjectFeatureQuery(o, f))
        get_image = lambda: self._send_query(self._image_query)
        return self._detector(predicates, objects, detect_feature, get_image)
