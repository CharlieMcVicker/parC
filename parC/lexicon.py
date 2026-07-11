from loguru import logger

from parC.constants import get_yaml_dir
from parC.yaml_utils.yaml_server import get_yaml_data_safe
import pandas as pd
import os
import numpy as np
from frozendict import frozendict


_lexicon_df_cache = {}

def load_lexicon_df(lexicon_basename: str) -> pd.DataFrame:
    if lexicon_basename not in _lexicon_df_cache:
        lexicon_dir = os.path.join(get_yaml_dir(), "Lexicon", "Wordlists")
        lexicon_stem = os.path.splitext(lexicon_basename.removeprefix("$"))[0]

        csv_path = os.path.join(lexicon_dir, lexicon_stem + ".csv")
        xlsx_path = os.path.join(lexicon_dir, lexicon_stem + ".xlsx")

        if os.path.exists(xlsx_path):
            df = pd.read_excel(xlsx_path, keep_default_na=False)
        elif os.path.exists(csv_path):
            df = pd.read_csv(csv_path, keep_default_na=False)
        else:
            init_lexicon(lexicon_basename, csv_path)
            df = pd.read_csv(csv_path, keep_default_na=False)
        _lexicon_df_cache[lexicon_basename] = df
    return _lexicon_df_cache[lexicon_basename].copy()


def init_lexicon(lexicon_basename: str, lexicon_path: str) -> None:
    part_of_speech = get_yaml_data_safe(
        yaml_basename=lexicon_basename, kind="PartOfSpeech"
    )
    lexical_features = part_of_speech.get("lexical_features", [])
    principal_parts = part_of_speech.get("principal_parts", [])
    df = pd.DataFrame(columns=["root", "gloss"] + lexical_features + principal_parts)
    os.makedirs(os.path.dirname(lexicon_path), exist_ok=True)
    df.to_csv(lexicon_path, index=False)


def get_roots(lexicon_basename: str) -> list[str]:
    lexicon_df = load_lexicon_df(lexicon_basename)
    return lexicon_df["root"].tolist()


def get_roots_with_lexical_features(
    lexicon_basename: str, lexical_features: set[tuple[str, str]] | dict[str, str]
) -> list[str]:
    if isinstance(lexical_features, (dict, frozendict)):
        lexical_features = set(lexical_features.items())
    df = load_lexicon_df(lexicon_basename)
    filter = pd.Series([True] * len(df), index=df.index)
    for feature, value in lexical_features:
        filter &= df[feature] == value

    filtered_df = df[filter]
    return filtered_df["root"].tolist()


def get_features_for_root(
    lexicon_basename: str, root: str
) -> tuple[tuple[str, str], ...]:
    df = load_lexicon_df(lexicon_basename)
    part_of_speech = get_yaml_data_safe(
        yaml_basename=lexicon_basename, kind="PartOfSpeech"
    )
    lexical_features = part_of_speech.get("lexical_features", [])
    row = df[df["root"] == root]
    if not row.empty:
        return tuple(row.iloc[0][lexical_features].items())
    return {}


def get_principal_part_for_root(
    lexicon_basename: str,
    root: str,
    principal_part: str,
    fallback_to_root: bool = True,
) -> str | None:
    df = load_lexicon_df(lexicon_basename)
    row = df[df["root"] == root]
    if not row.empty:
        principal_part = row.iloc[0][principal_part]
        if not principal_part and fallback_to_root:
            return root
        return principal_part
    logger.exception(f"Root not found: {root} in lexicon: {lexicon_basename}")
    return None


def get_principal_part_for_all_roots(
    lexicon_basename: str, principal_part: str, fallback_to_root: bool = True
) -> list[str]:
    df = load_lexicon_df(lexicon_basename)
    if fallback_to_root:
        return df[principal_part].replace("", np.nan).fillna(df["root"]).tolist()
    return df[principal_part].tolist()


def get_gloss_for_root(lexicon_basename: str, root: str) -> str | None:
    df = load_lexicon_df(lexicon_basename)
    row = df[df["root"] == root]
    if not row.empty:
        return row.iloc[0]["gloss"]
    return None


def get_roots_with_gloss(lexicon_basename: str, gloss: str) -> list[str]:
    df = load_lexicon_df(lexicon_basename)
    filtered_df = df[df["gloss"] == gloss]
    return filtered_df["root"].tolist()
