"""A modular perceiver."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    Hashable,
    Set,
    Tuple,
    TypeVar,
)

import numpy as np
from tomsutils.utils import draw_dag

Query = TypeVar("Query", bound=Hashable)
Response = TypeVar("Response")


class ModularPerceiver:
    """A modular perceiver."""

    def __init__(self, modules: Set[PerceptionModule]) -> None:
        self._modules = modules
        # Create connection from each module to this perceiver.
        for module in self._modules:
            module.set_perceiver(self)
        # For drawing and debugging, record the "connectivity" of the modules
        # in terms of which has ever sent a query to which.
        self._module_edges: Set[Tuple[PerceptionModule, PerceptionModule]] = set()

    def reset(self, seed: int | None = None) -> None:
        """Reset the modules of the perceiver."""
        for module in self._modules:
            module.reset(seed)

    def tick(self) -> None:
        """Advance time for all modules."""
        for module in self._modules:
            module.tick()

    def get_response(
        self,
        query: Hashable,
        sender: PerceptionModule | None = None,
    ) -> Any:
        """Find a module that can answer the query and generate a response.

        The sender is provided just for logging purposes. A sender of
        None means that the request came from outside the perceiver.
        """
        response = None
        responder: PerceptionModule | None = None
        for module in self._modules:
            try:
                response = module.get_response(query)
                assert responder is None, "Multiple modules can answer query"
                responder = module
            except ModuleCannotAnswerQuery:
                continue
        assert responder is not None, f"No module can answer query: {query}"
        if sender is not None:
            self._module_edges.add((responder, sender))
        return response

    def draw_connections(self, outfile: Path) -> None:
        """Draw the module connections based on queries sent so far."""
        edges = {
            (r.__class__.__name__, s.__class__.__name__) for r, s in self._module_edges
        }
        draw_dag(edges, outfile)


class ModuleCannotAnswerQuery(Exception):
    """Raised when a module is given a query it cannot answer."""


class PerceptionModule(abc.ABC, Generic[Query, Response]):
    """Base class for a module."""

    def __init__(self, seed: int = 0) -> None:
        self._time = 0
        self._query_to_response: Dict[Query, Response] = {}
        self._set_seed(seed)
        self._perceiver: ModularPerceiver | None = None

    def set_perceiver(self, perceiver: ModularPerceiver):
        """Set the perceiver for this module."""
        self._perceiver = perceiver

    def _set_seed(self, seed: int) -> None:
        """Set the internal random number generator."""
        self._rng = np.random.default_rng(seed)

    ################# Handling queries FROM other modules ####################

    @abc.abstractmethod
    def _get_response(self, query: Query) -> Response:
        """Module-specific logic for queries and responses."""
        raise NotImplementedError

    def get_response(self, query: Query) -> Response:
        """Answer a query and cache the response for this timestep."""
        try:
            return self._query_to_response[query]
        except KeyError:
            pass
        response = self._get_response(query)
        self._query_to_response[query] = response
        return response

    ################### Sending queries TO other modules ######################

    def _send_query(self, query: Hashable) -> Any:
        """Convenient method for querying another module.

        Note that the query type is not Query because it should be the
        parent's query type, not this module's query type. Same for
        response.
        """
        assert self._perceiver is not None
        return self._perceiver.get_response(query, self)

    ################### Managing internal state (memory) ######################

    def reset(self, seed: int | None = None) -> None:
        """Reset the module."""
        self._time = 0
        self._query_to_response = {}
        if seed is not None:
            self._set_seed(seed)

    def tick(self) -> None:
        """Advance time."""
        self._time += 1
        self._query_to_response = {}
