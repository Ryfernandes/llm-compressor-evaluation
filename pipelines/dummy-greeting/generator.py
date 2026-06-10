# Simple, dummy pipeline to demonstrate KFP functionality and verify pipeline execution

from typing import List, Optional

from kfp import dsl, compiler

PIPELINE_NAME = "dummy_greeting"

@dsl.component
def calculate_age(birth_year: int) -> int:
    age = 2026 - birth_year
    return age

@dsl.component
def construct_message(name: str, age: int) -> str:
    message = f"Hello, {name}! You are {age} years old."
    return message

@dsl.pipeline
def greeting_pipeline(name: str, birth_year: int) -> str:
    age_task = calculate_age(birth_year=birth_year)
    message_task = construct_message(name=name, age=age_task.output)
    return message_task.output

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=greeting_pipeline, package_path=f"{PIPELINE_NAME}.yaml"
    )