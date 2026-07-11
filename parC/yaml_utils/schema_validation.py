"""
Utilities for validating YAML data against JSON schemas.
"""

import json
from loguru import logger
from jsonschema import validate, ValidationError
from pathlib import Path
from parC.constants import SCHEMA_DIR
from typing import Literal, get_args
from frozendict import frozendict

ConfigKindType = Literal[
    # TODO: FeatureCombinations, MorphemeSet and MorphemeSequence are buggy
    # so they are commented out for now
    "ContingentFeatureMarkers",
    # "FeatureCombinations",
    "FeatureDefinitions",
    "FeatureMarkers",
    "Inventory",
    # "MorphemeSequence",
    # "MorphemeSet",
    "Paradigm",
    "PartOfSpeech",
    "Patterns",
    "Rules",
]
CONFIG_KINDS: tuple[str, ...] = get_args(ConfigKindType)

CONFIG_KIND_TO_PARDIR = {
    "ContingentFeatureMarkers": "Exponence",
    "FeatureDefinitions": "Exponence",
    "FeatureMarkers": "Exponence",
    "Inventory": "Phonology",
    "Rules": "Phonology",
    "Patterns": "Phonology",
    "Paradigm": "Morphotactics",
    "PartOfSpeech": "Lexicon",
    "Wordlists": "Lexicon",
}


def fix_refs_safe(schema: dict, SCHEMA_DIR: Path) -> dict:
    """
    Safely resolves external $ref in a JSON schema by copying the
    referenced content directly into the schema definitions and
    changing the $ref to a local reference. Avoids RecursionError
    caused by `jsonref` package.
    """

    node_stack = [schema]
    inserted_content = {}

    while node_stack:
        current_node = node_stack.pop()
        if isinstance(current_node, (dict, frozendict)):
            node_stack.extend(current_node.values())

            if "$ref" in current_node:
                ref_path = current_node["$ref"]
                if ref_path.startswith("#"):
                    continue  # Local reference, skip
                rel_path, object_path = ref_path.split("#")
                ref_file_path = SCHEMA_DIR / rel_path

                # get referenced content to be added to schema later
                content, object_name = get_referenced_content(
                    ref_file_path, object_path
                )
                inserted_content[object_name] = content

                # change $ref to local reference
                current_node["$ref"] = f"#{object_path}"

        elif isinstance(current_node, list):
            # lists may contain objects with $ref
            # so we need to check them as well
            node_stack.extend(current_node)

    # insert referenced content into schema
    if "definitions" not in schema:
        schema["definitions"] = {}
    for object_name, content in inserted_content.items():
        if object_name in schema["definitions"]:
            logger.warning(
                f"Object '{object_name}' already exists in schema definitions. Overwriting with content from {ref_file_path}."
            )
        schema["definitions"][object_name] = content
    return schema


def get_referenced_content(ref_file_path: Path, object_path: str) -> tuple[dict, str]:
    # JSON path should be in the format "definitions/{OBJECT_NAME}"
    path_parts = object_path.strip("/").split("/")
    assert (
        len(path_parts) == 2 and path_parts[0] == "definitions"
    ), "Only references to definitions are supported"
    definitions_key, object_name = path_parts

    if not ref_file_path.exists():
        raise FileNotFoundError(f"Referenced schema file not found: {ref_file_path}")

    with open(ref_file_path, "r", encoding="utf-8") as f:
        ref_content = json.load(f)
    content = ref_content.get(definitions_key, {}).get(object_name, None)
    if content is None:
        raise ValueError(
            f"Referenced object '{object_name}' not found in '{ref_file_path}'"
        )
    return content, object_name


def load_schema(target_kind: str):
    # Generate schema filename from kind
    schema_filename = f"{target_kind}.json"
    schema_path = Path(SCHEMA_DIR)
    schema_file_path = schema_path / schema_filename

    if not schema_file_path.exists():
        logger.error(f"Schema file not found: {schema_file_path}")
        return

    try:
        with open(schema_file_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        schema = fix_refs_safe(schema, schema_path)
        return schema
    except Exception as e:
        logger.exception(f"Failed to load schema {schema_filename}: {e}")
        return


def validate_yaml(target_kind: str, data: dict) -> dict:
    schema = load_schema(target_kind)
    try:
        validate(data, schema)
        return data
    except ValidationError as e:
        logger.exception(
            f"Failed to validate YAML data against schema {target_kind}: {e}"
        )
    except Exception as e:
        logger.exception(
            f"Non-schema related error occur while reading YAML data{target_kind}: {e}"
        )
