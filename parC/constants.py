import os
from dotenv import load_dotenv

# filepaths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

SCHEMA_DIR = os.path.join(PROJECT_ROOT, "schemas")

_YAML_DIR = None


def get_yaml_dir():
    global _YAML_DIR
    if not _YAML_DIR:
        _YAML_DIR = _get_yaml_dir()

    return _YAML_DIR


def _get_yaml_dir():
    load_dotenv(os.path.join(PROJECT_ROOT, "parC.env"))
    return os.environ.get("YAML_DIR") or os.path.join(
        PROJECT_ROOT, "yaml", "spanish-example"
    )


def set_yaml_dir(path: str):
    # TODO: add UI for changing YAML_DIR
    os.environ["YAML_DIR"] = path


# pynini constants


# copied from https://github.com/kylebgorman/pynini/blob/27ce19048193358cd362a4de6b157cb43ab6e2eb/extensions/stringcompile.h#L69
# a bit hacky: since ... TODO check if we really need to include EOS/BOS in symbol table
BOS_INDEX = 0xF8FE
EOS_INDEX = 0xF8FF
