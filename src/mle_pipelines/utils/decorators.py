from collections.abc import Callable
from typing import Any

from kfp import dsl

from mle_pipelines.utils.constants import get_hardware_constants, get_evaluation_constants

def hardware_component(
    *,
    additional_funcs: list[Callable] | None = None,
    **component_kwargs: Any
):
    merged_additional_funcs = [
        get_hardware_constants,
        *(additional_funcs or []),
    ]

    return dsl.component(
        additional_funcs=merged_additional_funcs,
        **component_kwargs,
    )

def evaluation_component(
    *,
    additional_funcs: list[Callable] | None = None,
    **component_kwargs: Any
):
    merged_additional_funcs = [
        get_evaluation_constants,
        *(additional_funcs or []),
    ]

    return dsl.component(
        additional_funcs=merged_additional_funcs,
        **component_kwargs,
    )
