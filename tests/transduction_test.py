from parC.yaml_utils.models import (
    Marker,
    Rule,
    SimpleRule,
    StringMapRule,
    RuleSequence,
    SingleStringMarker,
    StringTupleMarker,
    UnorderedMarker,
    PrincipalPartMarker,
    OperationTypeStringTuple,
    OperationTypeSingleString,
    UnorderedOperation,
)
from parC.grammar.transducer_compilation import compile_marker
from parC.grammar.acceptor_compilation import (
    fsa,
    word_fsa,
    fsm_strings,
    filter_strings_by_pattern,
)
from parC.grammar.marker_resolution import get_markers_for_paradigm
from parC.lexicon import get_roots_with_gloss
import pynini

from parC.yaml_utils.yaml_server import get_yaml_data_safe
from parC.grammar.paradigm_compilation import inflect, parse, search, _get_or_build

from parC.constants import PROJECT_ROOT
import os
import pytest


def test_suffix():
    marker = SingleStringMarker(kind="suffix", value="-sufijo")
    fst = compile_marker(marker)
    assert isinstance(fst, pynini.Fst)

    root = word_fsa("rama")
    result = pynini.compose(root, fst)
    assert result.num_states() > 0
    result_strings = fsm_strings(result, strip_all_tags=True)
    assert "rama-sufijo" in result_strings


def test_prefix():
    marker = SingleStringMarker(kind="prefix", value="antes-")
    fst = compile_marker(marker)
    assert isinstance(fst, pynini.Fst)

    root = word_fsa("historia")
    result = pynini.compose(root, fst)
    assert result.num_states() > 0
    result_strings = fsm_strings(result, strip_all_tags=True)
    assert "antes-historia" in result_strings


def test_rule():
    diphthongization_rule = "$diphthongization"
    marker = SingleStringMarker(kind="rule", value=diphthongization_rule)
    fst = compile_marker(marker)
    assert isinstance(fst, pynini.Fst)

    root = word_fsa("pod")
    result = pynini.compose(root, fst)
    assert result.num_states() > 0
    result_strings = fsm_strings(result, strip_all_tags=True)
    assert "pued" in result_strings


def test_2sg_a_class():

    # test fetching and applying markers manually

    feature_values = {
        "person_number": "2sg",
        "tense": "present",
        "mood": "indicative",
    }

    markers_2sg_a_class = get_markers_for_paradigm(
        feature_values=feature_values,
        paradigm_name="verb_a_stem",
    )

    assert len(markers_2sg_a_class) == 1
    marker = markers_2sg_a_class[0]
    assert isinstance(marker, SingleStringMarker)
    assert marker.kind == "suffix"
    assert marker.value == "-as"

    fst = compile_marker(marker)
    assert isinstance(fst, pynini.Fst)

    part_of_speech = get_yaml_data_safe(kind="Paradigm", yaml_basename="$verb_a_stem")[
        "part_of_speech"
    ]
    roots = get_roots_with_gloss(lexicon_basename=part_of_speech, gloss="speak")
    assert roots == ["habl"]

    root = roots[0]
    root_fsa = word_fsa(root)
    result = pynini.compose(root_fsa, fst)
    assert result.num_states() > 0
    result_strings = fsm_strings(result, strip_all_tags=True)
    expected_form = "habl-as"
    assert expected_form in result_strings

    # test inflection graph

    # here we invalidate the cache whenever the test is run
    # TODO: directly test automatic cache invalidation when source files are changed
    _get_or_build(graph_type="inflect", paradigm_name="verb_a_stem", force_rebuild=True)

    inflect_result = inflect(
        root=root, feature_values=feature_values, name="verb_a_stem"
    )
    assert inflect_result == result_strings

    _get_or_build(graph_type="parse", paradigm_name="verb_a_stem", force_rebuild=True)

    parse_result = parse(expected_form, kind="Paradigm", name="verb_a_stem")
    expected_parse = {
        "root": "habl",
        "gloss": "speak",
        "features": {"mood": "indicative", "tense": "present", "person_number": "2sg"},
    }
    assert len(parse_result) == 1
    assert parse_result[0] == expected_parse

    search_query = "habl-os"
    search_result = search(
        form=search_query, name="verb_a_stem", kind="Paradigm", nshortest=5
    )
    form_hits = [hit["form"] for hit in search_result]
    assert "habl-as" in form_hits
    assert "habl-o" in form_hits


def test_contingent_lexical_submapping():
    from parC.constants import get_yaml_dir
    from parC.yaml_utils.yaml_server import get_yaml_data_safe, get_feature_map
    import yaml

    yaml_dir = get_yaml_dir()
    contingent_dir = os.path.join(yaml_dir, "Exponence", "ContingentFeatureMarkers")
    paradigm_dir = os.path.join(yaml_dir, "Morphotactics", "Paradigm")

    contingent_path = os.path.join(contingent_dir, "class_test_contingent.yaml")
    paradigm_path = os.path.join(paradigm_dir, "verb_class_test.yaml")

    verb_pos = get_yaml_data_safe(kind="PartOfSpeech", yaml_basename="verb")
    lex_feats = verb_pos.get("lexical_features", [])
    pos_feats = verb_pos.get("features", [])

    if "conjugation_class" in lex_feats:
        class_name = "conjugation_class"
        feature = "person_number"
        class1, class2 = "a_class", "e_class"
        root1, root2 = "habl", "com"
        val1_1, val1_2 = "-o_a", "-as_a"
        val2_1, val2_2 = "-o_e", "-es_e"
        feat_markers = {"person_number": None, "tense": "present", "mood": "indicative"}
    elif "prefix_class" in lex_feats:
        class_name = "prefix_class"
        feature = "aspect" if "aspect" in pos_feats else "person_number"
        class1, class2 = "a_stem", "cons_stem"
        root1, root2 = "atvn", "woni"
        val1_1, val1_2 = "-o_a", "-as_a"
        val2_1, val2_2 = "-o_e", "-es_e"
        feat_markers = {feature: None}
        for f in pos_feats:
            if f != feature:
                fmap = get_feature_map()
                feat_markers[f] = fmap[f][0] if f in fmap and fmap[f] else None
    else:
        pytest.skip("No recognized lexical features for testing")
        return

    fmap = get_feature_map()
    feat_vals = fmap.get(feature, ["1sg", "2sg"])
    val_key1 = feat_vals[0]
    val_key2 = feat_vals[1] if len(feat_vals) > 1 else feat_vals[0]

    contingent_data = {
        "kind": "ContingentFeatureMarkers",
        "features": [class_name, feature],
        "markers": {
            class1: {
                val_key1: [{"kind": "suffix", "value": val1_1}],
                val_key2: [{"kind": "suffix", "value": val1_2}],
            },
            class2: {
                val_key1: [{"kind": "suffix", "value": val2_1}],
                val_key2: [{"kind": "suffix", "value": val2_2}],
            },
        },
    }

    paradigm_data = {
        "kind": "Paradigm",
        "part_of_speech": "$verb",
        "feature_markers": feat_markers,
        "contingent_markers": ["$class_test_contingent"],
    }

    os.makedirs(contingent_dir, exist_ok=True)
    os.makedirs(paradigm_dir, exist_ok=True)

    with open(contingent_path, "w", encoding="utf-8") as f:
        yaml.dump(contingent_data, f)

    with open(paradigm_path, "w", encoding="utf-8") as f:
        yaml.dump(paradigm_data, f)

    try:
        _get_or_build(
            graph_type="inflect", paradigm_name="verb_class_test", force_rebuild=True
        )

        # Test root1
        res_root1_1 = inflect(root1, {feature: val_key1}, "verb_class_test")
        assert f"{root1}{val1_1}" in res_root1_1

        res_root1_2 = inflect(root1, {feature: val_key2}, "verb_class_test")
        assert f"{root1}{val1_2}" in res_root1_2

        # Test root2
        res_root2_1 = inflect(root2, {feature: val_key1}, "verb_class_test")
        assert f"{root2}{val2_1}" in res_root2_1

        res_root2_2 = inflect(root2, {feature: val_key2}, "verb_class_test")
        assert f"{root2}{val2_2}" in res_root2_2

    finally:
        if os.path.exists(contingent_path):
            os.remove(contingent_path)
        if os.path.exists(paradigm_path):
            os.remove(paradigm_path)


def test_build_inflect_graph_for_root_regex():
    from parC.grammar.paradigm_compilation import build_inflect_graph_for_root_regex
    from parC.grammar.acceptor_compilation import word_fsa, fsa, fsm_strings
    from parC.fst_utils import stringify_features
    import pynini

    paradigm_name = "verb_a_stem"
    root_pattern = "cant"  # Represents a root string

    # Build the inflect graph using the root regex
    inflect_graph = build_inflect_graph_for_root_regex(paradigm_name, root_pattern)
    assert isinstance(inflect_graph, pynini.Fst)

    feature_values = {
        "person_number": "2sg",
        "tense": "present",
        "mood": "indicative",
    }
    feature_str = stringify_features(feature_values)

    input_fsa = pynini.concat(word_fsa("cant"), fsa(feature_str))
    output_lattice = pynini.compose(input_fsa, inflect_graph).optimize()
    output_lattice = pynini.project(output_lattice, project_type="output")
    surface_forms = fsm_strings(output_lattice, strip_all_tags=True)

    assert "cant-as" in surface_forms


def test_build_inflect_graph_for_root_regex_with_lexical_features():
    from parC.constants import get_yaml_dir
    from parC.yaml_utils.yaml_server import get_yaml_data_safe, get_feature_map
    from parC.grammar.paradigm_compilation import build_inflect_graph_for_root_regex
    from parC.grammar.acceptor_compilation import word_fsa, fsa, fsm_strings
    from parC.fst_utils import stringify_features
    import yaml
    import pynini

    yaml_dir = get_yaml_dir()
    contingent_dir = os.path.join(yaml_dir, "Exponence", "ContingentFeatureMarkers")
    paradigm_dir = os.path.join(yaml_dir, "Morphotactics", "Paradigm")

    contingent_path = os.path.join(contingent_dir, "regex_class_test_contingent.yaml")
    paradigm_path = os.path.join(paradigm_dir, "verb_regex_class_test.yaml")

    verb_pos = get_yaml_data_safe(kind="PartOfSpeech", yaml_basename="verb")
    lex_feats = verb_pos.get("lexical_features", [])
    pos_feats = verb_pos.get("features", [])

    if "conjugation_class" in lex_feats:
        class_name = "conjugation_class"
        feature = "person_number"
        class1, class2 = "a_class", "e_class"
        val1_1 = "-o_a"
        val2_1 = "-o_e"
        feat_markers = {"person_number": None, "tense": "present", "mood": "indicative"}
    elif "prefix_class" in lex_feats:
        class_name = "prefix_class"
        feature = "aspect" if "aspect" in pos_feats else "person_number"
        class1, class2 = "a_stem", "cons_stem"
        val1_1 = "-o_a"
        val2_1 = "-o_e"
        feat_markers = {feature: None}
        for f in pos_feats:
            if f != feature:
                fmap = get_feature_map()
                feat_markers[f] = fmap[f][0] if f in fmap and fmap[f] else None
    else:
        pytest.skip("No recognized lexical features for testing")
        return

    fmap = get_feature_map()
    feat_vals = fmap.get(feature, ["1sg", "2sg"])
    val_key1 = feat_vals[0]

    contingent_data = {
        "kind": "ContingentFeatureMarkers",
        "features": [class_name, feature],
        "markers": {
            class1: {
                val_key1: [{"kind": "suffix", "value": val1_1}],
            },
            class2: {
                val_key1: [{"kind": "suffix", "value": val2_1}],
            },
        },
    }

    paradigm_data = {
        "kind": "Paradigm",
        "part_of_speech": "$verb",
        "feature_markers": feat_markers,
        "contingent_markers": ["$regex_class_test_contingent"],
    }

    os.makedirs(contingent_dir, exist_ok=True)
    os.makedirs(paradigm_dir, exist_ok=True)

    with open(contingent_path, "w", encoding="utf-8") as f:
        yaml.dump(contingent_data, f)

    with open(paradigm_path, "w", encoding="utf-8") as f:
        yaml.dump(paradigm_data, f)

    try:
        # Build using root regex AND custom lexical features for class1
        inflect_graph_class1 = build_inflect_graph_for_root_regex(
            "verb_regex_class_test", "cant", lexical_features={class_name: class1}
        )
        assert isinstance(inflect_graph_class1, pynini.Fst)

        # Test inflection
        feature_values = {feature: val_key1}
        for f, v in feat_markers.items():
            if v is not None:
                feature_values[f] = v

        feature_str = stringify_features(feature_values)
        input_fsa = pynini.concat(word_fsa("cant"), fsa(feature_str))
        output_lattice = pynini.compose(input_fsa, inflect_graph_class1).optimize()
        output_lattice = pynini.project(output_lattice, project_type="output")
        surface_forms = fsm_strings(output_lattice, strip_all_tags=True)

        assert f"cant{val1_1}" in surface_forms

    finally:
        if os.path.exists(contingent_path):
            os.remove(contingent_path)
        if os.path.exists(paradigm_path):
            os.remove(paradigm_path)


def test_build_inflect_graph_for_root_regex_lexical_inference():
    from parC.constants import get_yaml_dir
    from parC.yaml_utils.yaml_server import get_yaml_data_safe, get_feature_map
    from parC.grammar.paradigm_compilation import build_inflect_graph_for_root_regex
    from parC.grammar.acceptor_compilation import word_fsa, fsm_strings
    import yaml
    import pynini

    yaml_dir = get_yaml_dir()
    contingent_dir = os.path.join(yaml_dir, "Exponence", "ContingentFeatureMarkers")
    paradigm_dir = os.path.join(yaml_dir, "Morphotactics", "Paradigm")

    contingent_path = os.path.join(contingent_dir, "inference_test_contingent.yaml")
    paradigm_path = os.path.join(paradigm_dir, "verb_inference_test.yaml")

    verb_pos = get_yaml_data_safe(kind="PartOfSpeech", yaml_basename="verb")
    lex_feats = verb_pos.get("lexical_features", [])
    pos_feats = verb_pos.get("features", [])

    if "conjugation_class" in lex_feats:
        class_name = "conjugation_class"
        feature = "person_number"
        class1, class2 = "a_class", "e_class"
        val1_1 = "-o_a"
        val2_1 = "-o_e"
        feat_markers = {"person_number": None, "tense": "present", "mood": "indicative"}
    elif "prefix_class" in lex_feats:
        class_name = "prefix_class"
        feature = "aspect" if "aspect" in pos_feats else "person_number"
        class1, class2 = "a_stem", "cons_stem"
        val1_1 = "-o_a"
        val2_1 = "-o_e"
        feat_markers = {feature: None}
        for f in pos_feats:
            if f != feature:
                fmap = get_feature_map()
                feat_markers[f] = fmap[f][0] if f in fmap and fmap[f] else None
    else:
        pytest.skip("No recognized lexical features for testing")
        return

    fmap = get_feature_map()
    feat_vals = fmap.get(feature, ["1sg", "2sg"])
    val_key1 = feat_vals[0]

    contingent_data = {
        "kind": "ContingentFeatureMarkers",
        "features": [class_name, feature],
        "markers": {
            class1: {
                val_key1: [{"kind": "suffix", "value": val1_1}],
            },
            class2: {
                val_key1: [{"kind": "suffix", "value": val2_1}],
            },
        },
    }

    paradigm_data = {
        "kind": "Paradigm",
        "part_of_speech": "$verb",
        "feature_markers": feat_markers,
        "contingent_markers": ["$inference_test_contingent"],
    }

    os.makedirs(contingent_dir, exist_ok=True)
    os.makedirs(paradigm_dir, exist_ok=True)

    with open(contingent_path, "w", encoding="utf-8") as f:
        yaml.dump(contingent_data, f)

    with open(paradigm_path, "w", encoding="utf-8") as f:
        yaml.dump(paradigm_data, f)

    try:
        # Build using root regex AND infer_lexical_features=True
        inflect_graph = build_inflect_graph_for_root_regex(
            "verb_inference_test", "cant", infer_lexical_features=True
        )
        assert isinstance(inflect_graph, pynini.Fst)

        # Invert to build a parse graph
        parse_graph = pynini.invert(inflect_graph).optimize()

        # Let's parse the surface form cant-o_a
        surface_fsa = word_fsa(f"cant{val1_1}")
        parse_lattice = pynini.compose(surface_fsa, parse_graph).optimize()
        parse_strs = fsm_strings(parse_lattice)

        # The parse string must map to: [BOW]cant[EOW][conjugation_class=a_class][person_number=1sg]...
        # Let's verify that class1 is present in the parse output strings!
        assert any(f"[{class_name}={class1}]" in s for s in parse_strs)
        # And class2 should NOT be in the parse output strings (since it corresponds to val2_1)
        assert not any(f"[{class_name}={class2}]" in s for s in parse_strs)

    finally:
        if os.path.exists(contingent_path):
            os.remove(contingent_path)
        if os.path.exists(paradigm_path):
            os.remove(paradigm_path)


def test_build_inflect_graph_for_root_regex_verb_a_stem():
    from parC.grammar.paradigm_compilation import build_inflect_graph_for_root_regex
    from parC.grammar.acceptor_compilation import word_fsa, fsm_strings, fsa
    from parC.fst_utils import stringify_features
    import pynini

    # Build the inflect graph for verb_a_stem, root "habl"
    inflect_graph = build_inflect_graph_for_root_regex("verb_a_stem", "habl")
    assert isinstance(inflect_graph, pynini.Fst)

    # Verify that a valid combination (2sg present indicative) inflects correctly to "habl-as"
    feature_values = {"person_number": "2sg", "tense": "present", "mood": "indicative"}
    feature_str = stringify_features(feature_values)
    input_fsa = pynini.concat(word_fsa("habl"), fsa(feature_str))

    output_lattice = pynini.compose(input_fsa, inflect_graph).optimize()
    output_lattice = pynini.project(output_lattice, project_type="output")
    surface_forms = fsm_strings(output_lattice, strip_all_tags=True)
    assert "habl-as" in surface_forms


def test_parse_with_inverted_open_root_graph():
    from parC.grammar.paradigm_compilation import build_inflect_graph_for_root_regex
    from parC.grammar.acceptor_compilation import word_fsa, fsm_strings
    import pynini

    # Build the inflect graph with "<Phone>*" as the open-ended root pattern
    inflect_graph = build_inflect_graph_for_root_regex("verb_a_stem", "<Phone>*")
    assert isinstance(inflect_graph, pynini.Fst)

    # Invert the graph to get the parse graph
    parse_graph = pynini.invert(inflect_graph).optimize()

    # Verify that "habl-as" parses back to "habl" with features [person_number=2sg][tense=present][mood=indicative]
    form_fsa_habl = word_fsa("habl-as")
    parse_lattice_habl = pynini.compose(form_fsa_habl, parse_graph).optimize()
    parse_strs_habl = fsm_strings(parse_lattice_habl)
    assert any(
        "habl" in s
        and "[person_number=2sg]" in s
        and "[tense=present]" in s
        and "[mood=indicative]" in s
        for s in parse_strs_habl
    )

    # Verify that "cant-as" parses back to "cant" with features [person_number=2sg][tense=present][mood=indicative]
    form_fsa_cant = word_fsa("cant-as")
    parse_lattice_cant = pynini.compose(form_fsa_cant, parse_graph).optimize()
    parse_strs_cant = fsm_strings(parse_lattice_cant)
    assert any(
        "cant" in s
        and "[person_number=2sg]" in s
        and "[tense=present]" in s
        and "[mood=indicative]" in s
        for s in parse_strs_cant
    )


def test_feature_value_acceptors():
    from parC.constants import get_yaml_dir
    from parC.yaml_utils.yaml_server import get_yaml_kind
    from parC.grammar.paradigm_compilation import build_inflect_graph_for_root_regex
    from parC.grammar.acceptor_compilation import get_feature_acceptor_fsts
    import yaml

    yaml_dir = get_yaml_dir()
    features_dir = os.path.join(yaml_dir, "Exponence", "FeatureDefinitions")
    pos_dir = os.path.join(yaml_dir, "Lexicon", "PartOfSpeech")
    paradigm_dir = os.path.join(yaml_dir, "Morphotactics", "Paradigm")

    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(paradigm_dir, exist_ok=True)

    feature_file = os.path.join(features_dir, "acceptor_test_features.yaml")
    pos_file = os.path.join(pos_dir, "acceptor_test_pos.yaml")
    markers_file = os.path.join(
        yaml_dir, "Exponence", "FeatureMarkers", "acceptor_test_markers.yaml"
    )
    paradigm_file = os.path.join(paradigm_dir, "acceptor_test_paradigm.yaml")

    feature_data = {
        "kind": "FeatureDefinitions",
        "features": {
            "acceptor_prefix_class": [
                "normal",
                {"name": "e_stem", "acceptor": "e<Phone>*"},
            ],
            "acceptor_person_number": ["1sg", "2sg"],
        },
    }

    pos_data = {
        "kind": "PartOfSpeech",
        "name": "acceptor_test_pos",
        "lexical_features": ["acceptor_prefix_class"],
        "features": ["acceptor_person_number"],
    }

    markers_data = {
        "kind": "FeatureMarkers",
        "feature": "acceptor_person_number",
        "markers": {
            "1sg": [{"kind": "suffix", "value": "-test"}],
            "2sg": [{"kind": "suffix", "value": "-test"}],
        },
    }

    paradigm_data = {
        "kind": "Paradigm",
        "part_of_speech": "$acceptor_test_pos",
        "feature_markers": {"acceptor_person_number": "$acceptor_test_markers"},
    }

    with open(feature_file, "w", encoding="utf-8") as f:
        yaml.dump(feature_data, f)
    with open(pos_file, "w", encoding="utf-8") as f:
        yaml.dump(pos_data, f)
    os.makedirs(os.path.dirname(markers_file), exist_ok=True)
    with open(markers_file, "w", encoding="utf-8") as f:
        yaml.dump(markers_data, f)
    with open(paradigm_file, "w", encoding="utf-8") as f:
        yaml.dump(paradigm_data, f)

    try:
        # Clear lru_cache and observed caches
        from parC.yaml_utils.yaml_server import _get_yaml_data_safe_cached
        from parC.grammar.acceptor_compilation import (
            get_feature_acceptor_fsts,
            get_pattern_fsts,
            get_symbol_table,
        )

        _get_yaml_data_safe_cached.cache_clear()
        get_feature_acceptor_fsts.cache_clear()
        get_pattern_fsts.cache_clear()
        get_symbol_table.cache_clear()

        # Let's verify feature acceptors compilation
        feature_acceptors = get_feature_acceptor_fsts()
        assert "acceptor_prefix_class=e_stem" in feature_acceptors

        # Test valid input (stem starting with e under e_stem constraint)
        # We can build root regex for root "evla" under lexical feature acceptor_prefix_class=e_stem
        inf_valid = build_inflect_graph_for_root_regex(
            "acceptor_test_paradigm",
            "evla",
            lexical_features={"acceptor_prefix_class": "e_stem"},
        )
        assert inf_valid.num_states() > 0

        # Test invalid input (stem starting with a under e_stem constraint)
        inf_invalid = build_inflect_graph_for_root_regex(
            "acceptor_test_paradigm",
            "avla",
            lexical_features={"acceptor_prefix_class": "e_stem"},
        )
        # Should be filtered out, meaning empty FST / 0 states
        assert inf_invalid.num_states() == 0

        # Test get_open_parse_graph with non_deterministic_cleanup and infer_lexical_features
        from parC.grammar.paradigm_compilation import get_open_parse_graph
        open_parse_graph = get_open_parse_graph(
            "acceptor_test_paradigm",
            non_deterministic_cleanup=True,
            infer_lexical_features=True,
        )
        assert isinstance(open_parse_graph, pynini.Fst)

        # Parse evla-test
        input_fsa = word_fsa("evla-test")
        output_lattice = pynini.compose(input_fsa, open_parse_graph).optimize()
        output_lattice = pynini.project(output_lattice, project_type="output")
        parses = fsm_strings(output_lattice, strip_all_tags=False)

        # Verify that the parsed string contains the inferred lexical feature [acceptor_prefix_class=e_stem]
        has_lexical = False
        for p in parses:
            if "[acceptor_prefix_class=e_stem]" in p:
                has_lexical = True
                break
        assert has_lexical, f"Expected parse to infer and contain lexical feature [acceptor_prefix_class=e_stem], got: {parses}"

    finally:
        # Clean up temporary test files
        csv_file = os.path.join(
            yaml_dir, "Lexicon", "Wordlists", "acceptor_test_pos.csv"
        )
        for p in [feature_file, pos_file, markers_file, paradigm_file, csv_file]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        from parC.yaml_utils.yaml_server import _get_yaml_data_safe_cached
        from parC.grammar.acceptor_compilation import (
            get_feature_acceptor_fsts,
            get_pattern_fsts,
            get_symbol_table,
        )

        _get_yaml_data_safe_cached.cache_clear()
        get_feature_acceptor_fsts.cache_clear()
        get_pattern_fsts.cache_clear()
        get_symbol_table.cache_clear()


def test_non_deterministic_cleanup():
    from parC.grammar.paradigm_compilation import build_inflect_graph_for_root_regex
    from parC.grammar.acceptor_compilation import word_fsa, fsa, fsm_strings
    from parC.fst_utils import stringify_features
    import pynini

    paradigm_name = "verb_a_stem"
    root_pattern = "cant"

    # 1. Compile with non_deterministic_cleanup=False (default)
    inflect_graph_det = build_inflect_graph_for_root_regex(
        paradigm_name, root_pattern, non_deterministic_cleanup=False
    )
    feature_values = {
        "person_number": "2sg",
        "tense": "present",
        "mood": "indicative",
    }
    feature_str = stringify_features(feature_values)
    input_fsa = pynini.concat(word_fsa("cant"), fsa(feature_str))

    output_det = pynini.compose(input_fsa, inflect_graph_det).optimize()
    output_det = pynini.project(output_det, project_type="output")
    det_strings = fsm_strings(output_det, strip_all_tags=False)
    for s in det_strings:
        assert "[person_number=2sg]" not in s
        assert "[tense=present]" not in s
        assert "[mood=indicative]" not in s

    # 2. Compile with non_deterministic_cleanup=True
    inflect_graph_nd = build_inflect_graph_for_root_regex(
        paradigm_name, root_pattern, non_deterministic_cleanup=True
    )
    output_nd = pynini.compose(input_fsa, inflect_graph_nd).optimize()
    output_nd = pynini.project(output_nd, project_type="output")
    nd_strings = fsm_strings(output_nd, strip_all_tags=False)

    has_tags = False
    for s in nd_strings:
        if "[person_number=2sg]" in s or "[tense=present]" in s or "[mood=indicative]" in s:
            has_tags = True
            break
    assert has_tags, f"Expected some output strings in {nd_strings} to retain tags when non-deterministic cleanup is active"

    has_fully_clean = False
    for s in nd_strings:
        if "[person_number=2sg]" not in s and "[tense=present]" not in s and "[mood=indicative]" not in s:
            has_fully_clean = True
            break
    assert has_fully_clean, f"Expected at least one fully cleaned string in {nd_strings}"


def test_parse_with_retained_tags():
    from parC.grammar.paradigm_compilation import get_open_parse_graph
    from parC.grammar.acceptor_compilation import word_fsa, fsa, fsm_strings
    import pynini

    paradigm_name = "verb_a_stem"

    # 1. Compile open parse graph with non_deterministic_cleanup=True
    open_parse_graph = get_open_parse_graph(paradigm_name, non_deterministic_cleanup=True)
    assert isinstance(open_parse_graph, pynini.Fst)

    # 2. Case A: Input is just the surface form "cant-as"
    input_fsa_no_tags = word_fsa("cant-as")
    output_lattice_no_tags = pynini.compose(input_fsa_no_tags, open_parse_graph).optimize()
    output_lattice_no_tags = pynini.project(output_lattice_no_tags, project_type="output")
    parses_no_tags = fsm_strings(output_lattice_no_tags, strip_all_tags=False)

    expected_parse = "[BOW]cant[EOW][mood=indicative][person_number=2sg][tense=present]"
    assert expected_parse in parses_no_tags

    # 3. Case B: Input has a retained feature tag: "cant-as [person_number=2sg]"
    input_fsa_with_tag = pynini.concat(word_fsa("cant-as"), fsa("[person_number=2sg]"))
    output_lattice_with_tag = pynini.compose(input_fsa_with_tag, open_parse_graph).optimize()
    output_lattice_with_tag = pynini.project(output_lattice_with_tag, project_type="output")
    parses_with_tag = fsm_strings(output_lattice_with_tag, strip_all_tags=False)

    assert expected_parse in parses_with_tag


def test_open_parse_with_inferred_lexical_features():
    from parC.grammar.paradigm_compilation import get_open_parse_graph
    import pynini

    paradigm_name = "verb_a_stem"
    open_parse_graph = get_open_parse_graph(
        paradigm_name, non_deterministic_cleanup=True, infer_lexical_features=True
    )
    assert isinstance(open_parse_graph, pynini.Fst)


def test_optional_feature_flags():
    from parC.constants import get_yaml_dir
    from parC.yaml_utils.yaml_server import get_yaml_kind
    from parC.grammar.paradigm_compilation import get_open_parse_graph, build_inflect_graph
    from parC.grammar.acceptor_compilation import word_fsa, fsm_strings
    import yaml
    import pandas as pd

    yaml_dir = get_yaml_dir()
    features_dir = os.path.join(yaml_dir, "Exponence", "FeatureDefinitions")
    pos_dir = os.path.join(yaml_dir, "Lexicon", "PartOfSpeech")
    markers_dir = os.path.join(yaml_dir, "Exponence", "FeatureMarkers")
    contingent_dir = os.path.join(yaml_dir, "Exponence", "ContingentFeatureMarkers")
    paradigm_dir = os.path.join(yaml_dir, "Morphotactics", "Paradigm")
    wordlist_dir = os.path.join(yaml_dir, "Lexicon", "Wordlists")

    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(markers_dir, exist_ok=True)
    os.makedirs(contingent_dir, exist_ok=True)
    os.makedirs(paradigm_dir, exist_ok=True)
    os.makedirs(wordlist_dir, exist_ok=True)

    feature_file = os.path.join(features_dir, "opt_test_features.yaml")
    pos_file = os.path.join(pos_dir, "opt_test_pos.yaml")
    markers_file = os.path.join(markers_dir, "opt_test_markers.yaml")
    contingent_file = os.path.join(contingent_dir, "opt_test_contingent.yaml")
    paradigm_file = os.path.join(paradigm_dir, "opt_test_paradigm.yaml")
    csv_file = os.path.join(wordlist_dir, "opt_test_pos.csv")

    feature_data = {
        "kind": "FeatureDefinitions",
        "features": {
            "is_irregular": {
                "values": ["true"],
                "optional": True,
            },
        },
    }

    pos_data = {
        "kind": "PartOfSpeech",
        "name": "opt_test_pos",
        "features": ["tense"],
        "lexical_features": ["is_irregular"],
    }

    markers_data = {
        "kind": "FeatureMarkers",
        "feature": "tense",
        "markers": {
            "past": [{"kind": "suffix", "value": "-a"}],
            "present": [{"kind": "suffix", "value": "-o"}],
        },
    }

    contingent_data = {
        "kind": "ContingentFeatureMarkers",
        "features": ["is_irregular", "tense"],
        "markers": {
            "true": {
                "past": [{"kind": "suffix", "value": "-e"}],
            }
        },
    }

    paradigm_data = {
        "kind": "Paradigm",
        "part_of_speech": "$opt_test_pos",
        "feature_markers": {
            "tense": "$opt_test_markers",
        },
        "contingent_markers": ["$opt_test_contingent"],
    }

    df = pd.DataFrame([
        {"root": "cant", "gloss": "sing", "is_irregular": ""},
        {"root": "evla", "gloss": "speak", "is_irregular": "true"},
    ])

    with open(feature_file, "w") as f:
        yaml.dump(feature_data, f)
    with open(pos_file, "w") as f:
        yaml.dump(pos_data, f)
    with open(markers_file, "w") as f:
        yaml.dump(markers_data, f)
    with open(contingent_file, "w") as f:
        yaml.dump(contingent_data, f)
    with open(paradigm_file, "w") as f:
        yaml.dump(paradigm_data, f)
    df.to_csv(csv_file, index=False)

    try:
        # Clear server caches so new yaml files are scanned
        from parC.yaml_utils.yaml_server import _get_yaml_data_safe_cached
        _get_yaml_data_safe_cached.cache_clear()
        
        from parC.grammar.paradigm_compilation import (
            _get_active_combos_for_paradigm,
            get_roots_for_paradigm
        )
        _get_active_combos_for_paradigm.cache_clear()
        get_roots_for_paradigm.cache_clear()
        
        from parC.grammar.marker_resolution import _feature_combos_for_paradigm_cache
        _feature_combos_for_paradigm_cache.clear()

        # Clear disk cache folder to force re-compilation of FSTs
        import shutil
        from parC.yaml_utils.cache import CACHE_DIR
        if os.path.exists(CACHE_DIR):
            try:
                shutil.rmtree(CACHE_DIR)
            except Exception:
                pass
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Build regular inflect graph (specific roots)
        inflect_graph = build_inflect_graph("opt_test_paradigm")
        assert isinstance(inflect_graph, pynini.Fst)

        # Build open parse graph with lexical feature inference
        open_parse_graph = get_open_parse_graph(
            "opt_test_paradigm", non_deterministic_cleanup=True, infer_lexical_features=True
        )
        assert isinstance(open_parse_graph, pynini.Fst)

        # 1. Parse cant-a -> should parse to cant[tense=past] (no is_irregular)
        input_fsa_walk = word_fsa("cant-a")
        output_lattice_walk = pynini.compose(input_fsa_walk, open_parse_graph).optimize()
        output_lattice_walk = pynini.project(output_lattice_walk, project_type="output")
        parses_walk = fsm_strings(output_lattice_walk, strip_all_tags=False)
        assert "[BOW]cant[EOW][tense=past]" in parses_walk

        # 2. Parse evla-e -> should parse to evla[is_irregular=true][tense=past]
        input_fsa_go = word_fsa("evla-e")
        output_lattice_go = pynini.compose(input_fsa_go, open_parse_graph).optimize()
        output_lattice_go = pynini.project(output_lattice_go, project_type="output")
        parses_go = fsm_strings(output_lattice_go, strip_all_tags=False)
        assert "[BOW]evla[EOW][is_irregular=true][tense=past]" in parses_go

    finally:
        # Clean up files
        for p in [feature_file, pos_file, markers_file, contingent_file, paradigm_file, csv_file]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


def test_optional_prefix_feature():
    from parC.constants import get_yaml_dir
    from parC.yaml_utils.yaml_server import get_yaml_kind
    from parC.grammar.paradigm_compilation import get_open_parse_graph, build_inflect_graph
    from parC.grammar.acceptor_compilation import word_fsa, fsm_strings
    import yaml
    import pandas as pd
    import pynini

    yaml_dir = get_yaml_dir()
    features_dir = os.path.join(yaml_dir, "Exponence", "FeatureDefinitions")
    pos_dir = os.path.join(yaml_dir, "Lexicon", "PartOfSpeech")
    markers_dir = os.path.join(yaml_dir, "Exponence", "FeatureMarkers")
    paradigm_dir = os.path.join(yaml_dir, "Morphotactics", "Paradigm")
    wordlist_dir = os.path.join(yaml_dir, "Lexicon", "Wordlists")

    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(markers_dir, exist_ok=True)
    os.makedirs(paradigm_dir, exist_ok=True)
    os.makedirs(wordlist_dir, exist_ok=True)

    feature_file = os.path.join(features_dir, "opt_prefix_features.yaml")
    pos_file = os.path.join(pos_dir, "opt_prefix_pos.yaml")
    markers_file = os.path.join(markers_dir, "opt_prefix_markers.yaml")
    tense_markers_file = os.path.join(markers_dir, "opt_tense_markers.yaml")
    paradigm_file = os.path.join(paradigm_dir, "opt_prefix_paradigm.yaml")
    csv_file = os.path.join(wordlist_dir, "opt_prefix_pos.csv")

    feature_data = {
        "kind": "FeatureDefinitions",
        "features": {
            "middle_voice": {
                "values": ["at", "atat", "ata"],
                "optional": True,
            },
        },
    }

    pos_data = {
        "kind": "PartOfSpeech",
        "name": "opt_prefix_pos",
        "features": ["tense"],
        "lexical_features": ["middle_voice"],
    }

    markers_data = {
        "kind": "FeatureMarkers",
        "feature": "middle_voice",
        "markers": {
            "at": [{"kind": "prefix", "value": "at-"}],
            "atat": [{"kind": "prefix", "value": "atat-"}],
            "ata": [{"kind": "prefix", "value": "ata-"}],
        },
    }

    tense_markers_data = {
        "kind": "FeatureMarkers",
        "feature": "tense",
        "markers": {
            "past": [{"kind": "suffix", "value": "-a"}],
        },
    }

    paradigm_data = {
        "kind": "Paradigm",
        "part_of_speech": "$opt_prefix_pos",
        "feature_markers": {
            "middle_voice": "$opt_prefix_markers",
            "tense": "$opt_tense_markers",
        },
    }

    df = pd.DataFrame([
        {"root": "atl", "gloss": "be", "middle_voice": ""},
        {"root": "atl", "gloss": "be", "middle_voice": "at"},
        {"root": "atl", "gloss": "be", "middle_voice": "atat"},
        {"root": "atl", "gloss": "be", "middle_voice": "ata"},
    ])

    with open(feature_file, "w") as f:
        yaml.dump(feature_data, f)
    with open(pos_file, "w") as f:
        yaml.dump(pos_data, f)
    with open(markers_file, "w") as f:
        yaml.dump(markers_data, f)
    with open(tense_markers_file, "w") as f:
        yaml.dump(tense_markers_data, f)
    with open(paradigm_file, "w") as f:
        yaml.dump(paradigm_data, f)
    df.to_csv(csv_file, index=False)

    try:
        from parC.yaml_utils.yaml_server import _get_yaml_data_safe_cached
        _get_yaml_data_safe_cached.cache_clear()
        
        from parC.grammar.paradigm_compilation import (
            _get_active_combos_for_paradigm,
            get_roots_for_paradigm
        )
        _get_active_combos_for_paradigm.cache_clear()
        get_roots_for_paradigm.cache_clear()
        
        from parC.grammar.marker_resolution import _feature_combos_for_paradigm_cache
        _feature_combos_for_paradigm_cache.clear()

        # Clear disk cache
        import shutil
        from parC.yaml_utils.cache import CACHE_DIR
        if os.path.exists(CACHE_DIR):
            try:
                shutil.rmtree(CACHE_DIR)
            except Exception:
                pass
        os.makedirs(CACHE_DIR, exist_ok=True)

        open_parse_graph = get_open_parse_graph(
            "opt_prefix_paradigm", non_deterministic_cleanup=True, infer_lexical_features=True
        )
        assert isinstance(open_parse_graph, pynini.Fst)

        # Parse atl-a (should parse to: [BOW]atl[EOW][tense=past])
        # It should NOT parse to [BOW]atl[EOW][middle_voice=at][tense=past] because at- was not present!
        input_fsa = word_fsa("atl-a")
        output_lattice = pynini.compose(input_fsa, open_parse_graph).optimize()
        output_lattice = pynini.project(output_lattice, project_type="output")
        parses = fsm_strings(output_lattice, strip_all_tags=False)

        print("DEBUG parses for atl-a:", parses)

        # Assertions
        assert "[BOW]atl[EOW][tense=past]" in parses
        for p in parses:
            if "[middle_voice=" in p:
                raise AssertionError(f"Prefix tag was not removed/enforced correctly: found '{p}' in parses: {parses}")

    finally:
        # Clean up files
        for p in [feature_file, pos_file, markers_file, tense_markers_file, paradigm_file, csv_file]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass




