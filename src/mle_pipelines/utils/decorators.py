from kfp import dsl

from mle_pipelines.utils.constants import get_hardware_constants, get_evaluation_constants

def hardware_component(**kwargs):
    return dsl.component(
        additional_funcs=[get_hardware_constants],
        **kwargs,
    )

def evaluation_component(**kwargs):
    return dsl.component(
        additional_funcs=[get_evaluation_constants],
        **kwargs,
    )
