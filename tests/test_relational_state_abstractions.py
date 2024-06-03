"""Example showing multiple levels of relational state abstractions."""

import numpy as np
from relational_structs import Predicate, Type

from modular_perception.modules.object_detection_module import ObjectDetectionModule
from modular_perception.modules.object_feature_module import ObjectFeatureModule
from modular_perception.modules.predicate_modules import (
    ImagePredicateModule,
    LocalPredicateModule,
    PredicateDispatchModule,
)
from modular_perception.modules.sensor_module import SensorModule
from modular_perception.perceiver import (
    ModularPerceiver,
)
from modular_perception.query_types import (
    AllGroundAtomsQuery,
    SensorQuery,
)


def test_relational_state_abstractions():
    """Example showing multiple levels of relational state abstractions."""

    # An "image" observation. The letters represent objects.
    observation = np.array(
        [
            ["X", "X", "X", "X", "X", "X", "X", "X", "G"],
            ["X", "A", "X", "X", "X", "X", "X", "X", "X"],
            ["X", "B", "X", "X", "C", "X", "X", "X", "X"],
            ["X", "X", "X", "X", "D", "X", "X", "X", "X"],
            ["X", "F", "X", "X", "X", "X", "X", "X", "X"],
            ["X", "X", "X", "X", "X", "X", "X", "X", "X"],
            ["X", "X", "X", "X", "X", "X", "E", "X", "X"],
            ["X", "X", "X", "X", "X", "X", "X", "X", "X"],
            ["X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ],
        dtype=object,
    )

    def _get_observation():
        # For this simple test, the observation is constant.
        return observation

    # Sensor module for getting the image.
    sensor_module = SensorModule({"camera": _get_observation})
    image_query = SensorQuery("camera")

    # Create an object detection module.
    Letter = Type("Letter")
    object_types = {Letter}

    # Detect all letters except for X and G.
    def _detect_objects(img, types):
        assert set(types) == object_types
        letters = set(np.unique(img)) - {"X", "G"}
        return frozenset(Letter(l) for l in letters)

    object_detection_module = ObjectDetectionModule(
        _detect_objects, sensory_input_query=image_query
    )

    # Create an object feature detection module.
    def _feature_detector(img, obj, feature):
        assert feature in ("r", "c")
        idxs = np.argwhere(img == obj.name)
        assert len(idxs) == 1
        r, c = idxs[0]
        if feature == "r":
            return r
        assert feature == "c"
        return c

    object_feature_module = ObjectFeatureModule(
        _feature_detector, sensory_input_query=image_query
    )

    # Create a "local" predicate classifier module. Local means using only
    # object-centric features.
    IsDirectlyAbove = Predicate("IsDirectlyAbove", [Letter, Letter])

    def _IsDirectlyAbove_holds(get_feature, obj1, obj2):
        r1 = get_feature(obj1, "r")
        c1 = get_feature(obj1, "c")
        r2 = get_feature(obj2, "r")
        c2 = get_feature(obj2, "c")
        return (r1 == r2 - 1) and (c1 == c2)

    IsAnywhereAbove = Predicate("IsAnywhereAbove", [Letter, Letter])

    def _IsAnywhereAbove_holds(get_feature, obj1, obj2):
        r1 = get_feature(obj1, "r")
        c1 = get_feature(obj1, "c")
        r2 = get_feature(obj2, "r")
        c2 = get_feature(obj2, "c")
        return (r1 < r2) and (c1 == c2)

    predicate_interpretations = {
        IsDirectlyAbove: _IsDirectlyAbove_holds,
        IsAnywhereAbove: _IsAnywhereAbove_holds,
    }
    local_predicate_module = LocalPredicateModule(
        predicate_interpretations,
    )

    # Define image-based predicates that use both object features and the
    # entire image.
    InOneThickEmptySpace = Predicate("InOneThickEmptySpace", [Letter])
    InTwoThickEmptySpace = Predicate("InTwoThickEmptySpace", [Letter])

    def _detect_image_predicates(predicates, objects, get_feature, get_image):
        # This could be implemented in a general way with a VLM instead.
        pred_to_pad = {InOneThickEmptySpace: 1, InTwoThickEmptySpace: 2}
        assert predicates.issubset(set(pred_to_pad))
        img = get_image()
        true_ground_atoms = set()

        def _has_empty_space(obj_r, obj_c, padding):
            for r in range(obj_r - padding, obj_r + padding + 1):
                if not 0 <= r < img.shape[0]:
                    continue
                for c in range(obj_c - padding, obj_c + padding + 1):
                    if not 0 <= c < img.shape[1]:
                        continue
                    if (r, c) == (obj_r, obj_c):
                        continue
                    if img[r, c] != "X":
                        return False
            return True

        for predicate in predicates:
            padding = pred_to_pad[predicate]
            for obj in objects:
                obj_r = int(get_feature(obj, "r"))
                obj_c = int(get_feature(obj, "c"))
                if _has_empty_space(obj_r, obj_c, padding):
                    ground_atom = predicate([obj])
                    true_ground_atoms.add(ground_atom)

        return true_ground_atoms

    image_predicate_module = ImagePredicateModule(
        _detect_image_predicates,
        image_query=image_query,
    )

    # We need to create a separate module that combines the two predicate
    # modules because there needs to be a mechanism for splitting up the query
    # into the respective predicate types.
    local_predicates = frozenset(predicate_interpretations)
    image_predicates = frozenset({InOneThickEmptySpace, InTwoThickEmptySpace})
    predicate_dispatch_module = PredicateDispatchModule(
        local_predicates=local_predicates,
        image_predicates=image_predicates,
        object_types=object_types,
    )

    # Finalize the perceiver.
    perceiver = ModularPerceiver(
        {
            sensor_module,
            object_feature_module,
            object_detection_module,
            local_predicate_module,
            image_predicate_module,
            predicate_dispatch_module,
        }
    )

    seed = 0
    perceiver.reset(seed)
    query = AllGroundAtomsQuery()
    result = perceiver.get_response(query)

    assert (
        str(sorted(result))
        == "[(InOneThickEmptySpace E), (InOneThickEmptySpace F), (InTwoThickEmptySpace E), (IsAnywhereAbove A B), (IsAnywhereAbove A F), (IsAnywhereAbove B F), (IsAnywhereAbove C D), (IsDirectlyAbove A B), (IsDirectlyAbove C D)]"  # pylint: disable=line-too-long
    )

    # Uncomment to plot.
    # from pathlib import Path
    # perceiver.draw_connections(Path("relational_state_abstraction_perceiver.pdf"))
