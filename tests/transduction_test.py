from parC.yaml_utils.models import (
    SingleStringMarker,
)
from parC.grammar.transducer_compilation import compile_marker
from parC.grammar.acceptor_compilation import (
    fsa,
    word_fsa,
    fsm_strings,
)
from parC.grammar.marker_resolution import get_markers_for_paradigm
from parC.lexicon import get_roots_with_gloss
import pynini

from parC.yaml_utils.yaml_server import get_yaml_data_safe
from parC.grammar.paradigm_compilation import inflect, parse, search, _get_or_build

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
                feat_markers[f] = fmap[f][0] if f in fmap and fmap[f] else "unmarked"
    else:
        pytest.skip("No recognized lexical features for testing")
        return

    fmap = get_feature_map()
    feat_vals = fmap.get(feature, ["1sg", "2sg"])
    val_key1 = feat_vals[0]
    val_key2 = feat_vals[1] if len(feat_vals) > 1 else "unmarked"

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
    from parC.grammar.acceptor_compilation import word_fsa, fsm_strings
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
    from parC.grammar.acceptor_compilation import word_fsa, fsm_strings
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
                feat_markers[f] = fmap[f][0] if f in fmap and fmap[f] else "unmarked"
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
                feat_markers[f] = fmap[f][0] if f in fmap and fmap[f] else "unmarked"
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


def test_get_label_to_marker_fst():
    from parC.grammar.paradigm_compilation import get_label_to_marker_fst
    from parC.grammar.acceptor_compilation import word_fsa, fsm_strings, get_symbol_table
    import pynini

    # 1. Compile the label-to-marker fst (exponence transducer) for verb_a_stem
    exponence_fst = get_label_to_marker_fst("verb_a_stem", infer_lexical_features=False)
    assert isinstance(exponence_fst, pynini.Fst)

    # 2. Let's test a mapping: habl[person_number=2sg][tense=present][mood=indicative]
    # should map to habl followed by its operational tag
    syms = get_symbol_table()
    input_fsa = word_fsa("habl")
    input_fsa = pynini.concat(input_fsa, pynini.accep("[mood=indicative]", token_type=syms))
    input_fsa = pynini.concat(input_fsa, pynini.accep("[person_number=2sg]", token_type=syms))
    input_fsa = pynini.concat(input_fsa, pynini.accep("[tense=present]", token_type=syms))

    result = pynini.compose(input_fsa, exponence_fst)
    assert result.num_states() > 0
    result_strs = fsm_strings(result, strip_all_tags=False)
    assert len(result_strs) == 1
    assert "[OP=suffix_as]" in result_strs[0]


def test_get_stage_realization_fst():
    from parC.grammar.paradigm_compilation import get_stage_realization_fst
    from parC.grammar.acceptor_compilation import word_fsa, fsm_strings, get_symbol_table
    import pynini

    # 1. Compile the stage-by-stage realization transducer for verb_a_stem, stage None
    stage_fst = get_stage_realization_fst("verb_a_stem", stage=None)
    assert isinstance(stage_fst, pynini.Fst)

    # 2. Let's test a mapping: habl[OP=suffix_as] should map to habl-as
    syms = get_symbol_table()
    input_fsa = word_fsa("habl")
    input_fsa = pynini.concat(input_fsa, pynini.accep("[OP=suffix_as]", token_type=syms))

    result = pynini.compose(input_fsa, stage_fst)
    assert result.num_states() > 0
    result_strs = fsm_strings(result, strip_all_tags=True)
    # The operational tag should be deleted/rewritten, and suffix -as appended
    assert "habl-as" in result_strs

    # 3. Test non-flagged identity mapping: a string with no active stage tags should pass through unchanged
    input_no_tag = word_fsa("habl")
    result_no_tag = pynini.compose(input_no_tag, stage_fst)
    assert result_no_tag.num_states() > 0
    result_no_tag_strs = fsm_strings(result_no_tag, strip_all_tags=True)
    assert "habl" in result_no_tag_strs


def test_get_final_surface_filter_fst():
    from parC.grammar.paradigm_compilation import get_final_surface_filter_fst
    from parC.grammar.acceptor_compilation import word_fsa, get_symbol_table

    # 1. Compile the final surface filter fst
    filter_fst = get_final_surface_filter_fst("verb_a_stem")
    assert isinstance(filter_fst, pynini.Fst)

    # 2. Check that valid strings are accepted
    assert pynini.compose(word_fsa("habl-as"), filter_fst).num_states() > 0
    assert pynini.compose(word_fsa("cant-as"), filter_fst).num_states() > 0

    # 3. Check that strings with remaining tags are rejected
    syms = get_symbol_table()
    
    # habl-as[OP=suffix_as]
    tagged_1 = pynini.concat(word_fsa("habl-as"), pynini.accep("[OP=suffix_as]", token_type=syms))
    assert pynini.compose(tagged_1, filter_fst).num_states() == 0

    # habl[mood=indicative]
    tagged_2 = pynini.concat(word_fsa("habl"), pynini.accep("[mood=indicative]", token_type=syms))
    assert pynini.compose(tagged_2, filter_fst).num_states() == 0


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






