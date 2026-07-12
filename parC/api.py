"""
FastAPI backend for parC grammar.
Provides following endpoints:
- `GET /schema/<kind>`: Retrieve the schema for a specific configuration kind.
- `GET /configs/<kind>`: List all configuration files for a specific kind.
- `GET /file/<kind>/<path>`: Read a specific configuration file.
- `PUT /file/<kind>/<path>`: Update a specific configuration file.
"""

import os
import pynini
from fastapi import FastAPI, HTTPException
from loguru import logger
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pynini.lib import rewrite
import uvicorn

from parC.grammar.acceptor_compilation import (
    fsa,
    word_fsa,
    fsm_strings,
    get_symbol_table,
)
from parC.grammar.paradigm_compilation import (
    inflect_stages,
    parse,
    search,
    get_roots_for_paradigm,
)
from parC.grammar.transducer_compilation import get_rule_fst
from parC.grammar.marker_resolution import get_free_features_for_paradigm
from parC.yaml_utils.yaml_server import (
    get_yaml_kind,
    get_inventory_items,
    get_feature_map,
    get_patterns,
    get_rules,
    get_inflection_stages,
    get_yaml_data_safe,
)
from parC.lexicon import (
    get_roots,
    get_features_for_root,
)

app = FastAPI()


def run_app():
    uvicorn.run("parC.api:app", host="127.0.0.1", port=8000, reload=True)


@app.get("/grammar-stats")
def grammar_stats() -> dict:

    grammar_stats = {}

    inventory_stats = {}
    inventory_items = get_inventory_items()
    inventory_yaml = get_yaml_kind("Inventory")
    inventory_stats["files"] = len(inventory_yaml["valid"])
    inventory_stats["invalid_files"] = len(inventory_yaml["invalid"])
    inventory_stats["phones"] = len(inventory_items.phones)
    inventory_stats["tags"] = len(inventory_items.tags)
    inventory_stats["classes"] = len(inventory_items.item_map)
    grammar_stats["inventory"] = inventory_stats

    feature_definitions_stats = {}
    feature_definitions_yaml = get_yaml_kind("FeatureDefinitions")
    features = get_feature_map()
    feature_definitions_stats["files"] = len(feature_definitions_yaml["valid"])
    feature_definitions_stats["invalid_files"] = len(
        feature_definitions_yaml["invalid"]
    )
    feature_definitions_stats["total"] = len(features)
    grammar_stats["feature_definitions"] = feature_definitions_stats

    feature_markers_stats = {}
    feature_markers_yaml = get_yaml_kind("FeatureMarkers")
    feature_markers_stats["files"] = len(feature_markers_yaml["valid"])
    feature_markers_stats["invalid_files"] = len(feature_markers_yaml["invalid"])
    feature_markers_stats["total"] = sum(
        len(file["markers"]) for _, file in feature_markers_yaml["valid"]
    )
    feature_markers_stats["inflection_stages"] = len(get_inflection_stages())
    grammar_stats["feature_markers"] = feature_markers_stats

    contingent_markers_stats = {}
    contingent_markers_yaml = get_yaml_kind("ContingentFeatureMarkers")
    contingent_markers_stats["files"] = len(contingent_markers_yaml["valid"])
    contingent_markers_stats["invalid_files"] = len(contingent_markers_yaml["invalid"])
    contingent_markers_stats["total"] = sum(
        len(file["markers"]) for _, file in contingent_markers_yaml["valid"]
    )
    grammar_stats["contingent_markers"] = contingent_markers_stats

    patterns_stats = {}
    patterns_yaml = get_yaml_kind("Patterns")
    patterns = get_patterns()
    patterns_stats["files"] = len(patterns_yaml["valid"])
    patterns_stats["invalid_files"] = len(patterns_yaml["invalid"])
    patterns_stats["total"] = len(patterns)
    grammar_stats["patterns"] = patterns_stats

    rules_stats = {}
    rules_yaml = get_yaml_kind("Rules")
    rules = get_rules()
    rules_stats["files"] = len(rules_yaml["valid"])
    rules_stats["invalid_files"] = len(rules_yaml["invalid"])
    rules_stats["total"] = len(rules)
    grammar_stats["rules"] = rules_stats

    paradigm_stats = {}
    paradigm_yaml = get_yaml_kind("Paradigm")
    paradigm_stats["files"] = len(paradigm_yaml["valid"])
    paradigm_stats["invalid_files"] = len(paradigm_yaml["invalid"])
    grammar_stats["paradigms"] = paradigm_stats

    part_of_speech_stats = {}
    part_of_speech_yaml = get_yaml_kind("PartOfSpeech")
    part_of_speech_stats["files"] = len(part_of_speech_yaml["valid"])
    num_roots = 0
    for part_of_speech, _ in part_of_speech_yaml["valid"]:
        current_roots = get_roots(part_of_speech)
        num_roots += len(current_roots)
    part_of_speech_stats["roots"] = num_roots
    part_of_speech_stats["invalid_files"] = len(part_of_speech_yaml["invalid"])
    grammar_stats["part_of_speech"] = part_of_speech_stats

    return grammar_stats


@app.get("/health")
def health_check():
    return {"status": "healthy"}


class InflectRequest(BaseModel):
    kind: str = "Paradigm"
    name: str
    root: str
    features: dict[str, str]


class ParseRequest(BaseModel):
    kind: str = "Paradigm"
    name: str
    form: str
    open_ended: bool = False


class ParadigmInfo(BaseModel):
    name: str
    features: list[str]
    lexical_features: list[str]


class InflectMeta(BaseModel):
    features: dict[str, tuple[str, ...]]
    paradigms: list[ParadigmInfo]


@app.get("/inflection-meta")
def inflection_meta():
    feature_map = get_feature_map()

    paradigms = []
    paradigm_yaml = get_yaml_kind("Paradigm")
    for basename, data in paradigm_yaml["valid"]:
        part_of_speech = get_yaml_data_safe(
            yaml_basename=data["part_of_speech"], kind="PartOfSpeech"
        )
        paradigms.append(
            ParadigmInfo(
                name=os.path.splitext(basename)[0],
                features=get_free_features_for_paradigm(basename, "Paradigm"),
                lexical_features=part_of_speech.get("lexical_features", []),
            )
        )

    return InflectMeta(features=feature_map, paradigms=paradigms)


@app.get("/roots")
def get_roots_route(kind: str, name: str):
    """
    Get the roots of a paradigm or lexicon.
    """
    if kind == "Paradigm":
        return get_roots_for_paradigm(name)
    elif kind == "Lexicon":
        return get_roots(name)
    else:
        logger.exception(f"Invalid kind: {kind}")
        raise HTTPException(status_code=400, detail="Invalid kind")


@app.get("/lexical-features")
def get_lexical_features(kind: str, name: str, root: str):
    """
    Get the lexical features of a single root in a paradigm or lexicon.
    """
    if kind == "Paradigm":
        paradigm = get_yaml_data_safe(kind="Paradigm", yaml_basename=name)
        part_of_speech_name = paradigm.get("part_of_speech")
        return get_features_for_root(part_of_speech_name, root)
    elif kind == "Lexicon":
        return get_features_for_root(name, root)
    else:
        logger.exception(f"Invalid kind: {kind}")
        raise HTTPException(status_code=400, detail="Invalid kind")


@app.get("/patterns")
def get_patterns_route():
    return [
        {
            "ref": ref,
            "value": pat.pattern,
            "test_includes": list(pat.test_includes or []),
            "test_excludes": list(pat.test_excludes or []),
        }
        for ref, pat in get_patterns().items()
    ]


@app.get("/rules")
def get_rules_route():
    return [{"name": name, **rule._asdict()} for name, rule in get_rules().items()]


class TestPatternRequest(BaseModel):
    pattern: str
    test_includes: list[str] = []
    test_excludes: list[str] = []


@app.post("/test-pattern")
def api_test_pattern(req: TestPatternRequest):
    try:
        pattern_fst = fsa(req.pattern)
        results = []
        all_pass = True
        for test_str in req.test_includes:
            intersection = pynini.intersect(pattern_fst, fsa(test_str))
            passed = intersection.start() != pynini.NO_STATE_ID
            results.append({"string": test_str, "test_kind": "include", "pass": passed})
            if not passed:
                all_pass = False
        for test_str in req.test_excludes:
            intersection = pynini.intersect(pattern_fst, fsa(test_str))
            passed = intersection.start() == pynini.NO_STATE_ID
            results.append({"string": test_str, "test_kind": "exclude", "pass": passed})
            if not passed:
                all_pass = False
        result = {"ref": req.pattern, "results": results, "all_pass": all_pass}
    except Exception as e:
        logger.exception(f"Error during pattern testing: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    return result


class TestRuleRequest(BaseModel):
    rule: str
    test_mappings: list[list[str]]


@app.post("/test-rule")
def api_test_rule(req: TestRuleRequest):
    try:
        rule_fst = get_rule_fst(req.rule)
        syms = get_symbol_table()
        results = []
        all_pass = True
        for input_str, expected_output_str in req.test_mappings:
            input_fsa = word_fsa(input_str)
            if isinstance(rule_fst, list):
                output_fst = input_fsa
                for sub_fst in rule_fst:
                    output_fst = rewrite.rewrite_lattice(
                        string=output_fst, rule=sub_fst, token_type=syms
                    )
                    output_fst.optimize()
            else:
                output_fst = rewrite.rewrite_lattice(
                    string=input_fsa, rule=rule_fst, token_type=syms
                )
                output_fst.optimize()
            output_projected = pynini.project(output_fst, project_type="output")
            passed = (
                pynini.intersect(
                    output_projected, word_fsa(expected_output_str)
                ).start()
                != pynini.NO_STATE_ID
            )
            output_strs = fsm_strings(output_projected, strip_all_tags=True)
            results.append(
                {
                    "input": input_str,
                    "output": output_strs,
                    "expected_output": expected_output_str,
                    "pass": passed,
                }
            )
            if not passed:
                all_pass = False
        result = {"ref": req.rule, "results": results, "all_pass": all_pass}
    except Exception as e:
        logger.exception(f"Error during rule testing: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    return result


class SearchRequest(BaseModel):
    kind: str = "Paradigm"
    name: str
    form: str
    nshortest: int = 10
    open_ended: bool = False


@app.post("/inflect")
def api_inflect(req: InflectRequest):
    try:
        stages = inflect_stages(
            root=req.root, feature_values=req.features, name=req.name
        )
        forms = stages[-1].surface_forms
        stages = [stage._asdict() for stage in stages]
    except Exception as e:
        logger.exception(f"Error during inflection: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    result = {"forms": forms, "stages": stages}
    logger.info(f"Inflect JSON: {result}")
    return result


@app.post("/parse")
def api_parse(req: ParseRequest):
    try:
        parses = parse(form=req.form, kind=req.kind, name=req.name, open_ended=req.open_ended)
    except Exception as e:
        logger.exception(f"Error during parsing: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    return {"parses": parses}


@app.post("/search")
def api_search(req: SearchRequest):
    try:
        parsed_hits = search(
            name=req.name,
            form=req.form,
            nshortest=req.nshortest,
            do_parse=True,
            kind=req.kind,
            open_ended=req.open_ended,
        )
    except Exception as e:
        logger.exception(f"Error during search: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    return {"parses": parsed_hits}


app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
