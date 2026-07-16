import os
import shutil
import yaml
import pandas as pd
import pytest
import pynini

from parC.constants import get_yaml_dir
from parC.yaml_utils.yaml_server import _get_yaml_data_safe_cached
from parC.grammar.paradigm_compilation import (
    parse,
    _get_or_build,
    get_inflect_graph,
    _get_active_combos_for_paradigm,
    get_roots_for_paradigm,
)
from parC.grammar.marker_resolution import _feature_combos_for_paradigm_cache
from parC.grammar.acceptor_compilation import (
    fsa,
    word_fsa,
    fsm_strings,
    get_pattern_fsts,
    get_special_fsas,
    get_feature_acceptor_fsts,
    get_symbol_table,
)
from parC.yaml_utils.cache import CACHE_DIR


def test_optional_feature_rule_swap():
    """
    Test an optional morphological feature '+' realized with a rule.
    When 'vowel_swap' is set to '+', the rule swap_a_to_o is triggered, changing 'a' to 'o'.
    When 'vowel_swap' is absent (since it is optional), no rule is triggered, leaving 'a' unchanged.
    """
    yaml_dir = get_yaml_dir()
    
    # Define directories
    features_dir = os.path.join(yaml_dir, "Exponence", "FeatureDefinitions")
    pos_dir = os.path.join(yaml_dir, "Lexicon", "PartOfSpeech")
    rules_dir = os.path.join(yaml_dir, "Phonology", "Rules")
    markers_dir = os.path.join(yaml_dir, "Exponence", "FeatureMarkers")
    paradigm_dir = os.path.join(yaml_dir, "Morphotactics", "Paradigm")
    wordlist_dir = os.path.join(yaml_dir, "Lexicon", "Wordlists")

    # Ensure all directories exist
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(rules_dir, exist_ok=True)
    os.makedirs(markers_dir, exist_ok=True)
    os.makedirs(paradigm_dir, exist_ok=True)
    os.makedirs(wordlist_dir, exist_ok=True)

    # File paths
    feature_file = os.path.join(features_dir, "opt_rule_features.yaml")
    pos_file = os.path.join(pos_dir, "opt_rule_pos.yaml")
    rules_file = os.path.join(rules_dir, "opt_rule_phonology.yaml")
    markers_file = os.path.join(markers_dir, "opt_rule_markers.yaml")
    paradigm_file = os.path.join(paradigm_dir, "opt_rule_paradigm.yaml")
    csv_file = os.path.join(wordlist_dir, "opt_rule_pos.csv")

    # Define minimal grammar YAML payloads
    feature_data = {
        "kind": "FeatureDefinitions",
        "features": {
            "vowel_swap": {
                "values": ["+"],
                "optional": True,
            },
        },
    }

    pos_data = {
        "kind": "PartOfSpeech",
        "name": "opt_rule_pos",
        "features": ["vowel_swap"],
    }

    rules_data = {
        "kind": "Rules",
        "rules": [
            {
                "name": "swap_a_to_o",
                "input_pattern": "a",
                "output_pattern": "o",
            }
        ],
    }

    markers_data = {
        "kind": "FeatureMarkers",
        "feature": "vowel_swap",
        "markers": {
            "+": [
                {
                    "kind": "rule",
                    "value": "$swap_a_to_o",
                }
            ],
        },
    }

    paradigm_data = {
        "kind": "Paradigm",
        "part_of_speech": "$opt_rule_pos",
        "feature_markers": {
            "vowel_swap": "$opt_rule_markers",
        },
    }

    # Wordlist with a test verb root containing 'a'
    df_roots = pd.DataFrame([
        {"root": "cant", "gloss": "sing"},
    ])

    # Write configs to files
    with open(feature_file, "w") as f:
        yaml.dump(feature_data, f)
    with open(pos_file, "w") as f:
        yaml.dump(pos_data, f)
    with open(rules_file, "w") as f:
        yaml.dump(rules_data, f)
    with open(markers_file, "w") as f:
        yaml.dump(markers_data, f)
    with open(paradigm_file, "w") as f:
        yaml.dump(paradigm_data, f)
    df_roots.to_csv(csv_file, index=False)

    try:
        # Clear server and compiler caches to register new/modified configuration
        _get_yaml_data_safe_cached.cache_clear()
        _get_active_combos_for_paradigm.cache_clear()
        get_roots_for_paradigm.cache_clear()
        _feature_combos_for_paradigm_cache.clear()
        get_pattern_fsts.cache_clear()
        get_special_fsas.cache_clear()
        get_feature_acceptor_fsts.cache_clear()
        get_symbol_table.cache_clear()

        # Clear physical disk cache to force FST recompilation
        if os.path.exists(CACHE_DIR):
            try:
                shutil.rmtree(CACHE_DIR)
            except Exception:
                pass
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Build / re-compile FST compilation graphs for this paradigm
        _get_or_build(graph_type="inflect", paradigm_name="opt_rule_paradigm", force_rebuild=True)
        _get_or_build(graph_type="parse", paradigm_name="opt_rule_paradigm", force_rebuild=True)

        # Retrieve Compiled Inflection Graph
        inflect_graph = get_inflect_graph("opt_rule_paradigm")

        # Create test cases dataframe
        # - Case 1: feature is absent (should inflect to 'cant', parse back to empty features)
        # - Case 2: feature is '+' (should inflect to 'cont' by swapping a->o, parse back to vowel_swap='+')
        df_test_cases = pd.DataFrame([
            {
                "root": "cant",
                "features": {},
                "expected_inflection": "cant",
            },
            {
                "root": "cant",
                "features": {"vowel_swap": "+"},
                "expected_inflection": "cont",
            },
        ])

        # Test all cases in the dataframe
        for _, row in df_test_cases.iterrows():
            root_val = row["root"]
            feats_val = row["features"]
            expected_infl = row["expected_inflection"]

            # Manual Inflection / Generation
            if feats_val:
                # If feature is present, concatenate root FSA with the feature tag FSA
                feat_str = f"[vowel_swap=+]"
                input_fsa = pynini.concat(word_fsa(root_val), fsa(feat_str))
            else:
                # If feature is absent/optional, the input is just the root word FSA
                input_fsa = word_fsa(root_val)

            output_lattice = pynini.compose(input_fsa, inflect_graph).optimize()
            output_lattice = pynini.project(output_lattice, project_type="output")
            inflection_result = fsm_strings(output_lattice, strip_all_tags=True)

            # Expecting the resulting surface form
            assert expected_infl in inflection_result, f"Expected {expected_infl} in inflection results {inflection_result}"

            # Test Parsing (Analysis)
            parse_result = parse(
                expected_infl,
                kind="Paradigm",
                name="opt_rule_paradigm",
            )
            
            # Verify we get a valid parse matching our input parameters
            found_matching_parse = False
            for parse_entry in parse_result:
                if parse_entry["root"] == root_val and parse_entry["features"] == feats_val:
                    found_matching_parse = True
                    break
            
            assert found_matching_parse, f"Could not find matching parse for {expected_infl} with features {feats_val} in {parse_result}"

    finally:
        # Clean up files from the yaml directory
        files_to_remove = [feature_file, pos_file, rules_file, markers_file, paradigm_file, csv_file]
        for p in files_to_remove:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
