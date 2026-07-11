import os

import pynini
from loguru import logger

from parC.constants import get_yaml_dir
from functools import lru_cache, wraps
from glob import glob
from frozendict import frozendict

CACHE_DIR = os.path.join(get_yaml_dir(), ".cache")
_SYMS_PATH = os.path.join(CACHE_DIR, "symbol_table.syms")


def _fst_path(kind: str, name: str, fst_kind: str) -> str:
    return os.path.join(CACHE_DIR, kind, f"{name}.{fst_kind}.fst")


def _is_valid(path: str, *source_dirs: str) -> bool:

    if not os.path.exists(path):
        return False
    mtime = os.path.getmtime(path)
    glob_list = []
    for source_dir in source_dirs:
        glob_list.extend(glob(os.path.join(source_dir, "*.yaml")))
        glob_list.extend(glob(os.path.join(source_dir, "*.csv")))
    return all(mtime >= os.path.getmtime(file) for file in glob_list)


def is_syms_cache_valid(*source_dirs: str) -> bool:
    if _is_valid(_SYMS_PATH, *source_dirs):
        return True
    logger.info("Symbol table cache invalidated.")
    return False


def is_fst_cache_valid(kind: str, name: str, fst_kind: str, *source_dirs: str) -> bool:
    if _is_valid(_fst_path(kind, name, fst_kind), *source_dirs):
        return True
    logger.info(f"Fst of kind {kind} {fst_kind} for {name} invalidated.")


def save_symbol_table(syms: pynini.SymbolTable) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    syms.write(_SYMS_PATH)


def load_symbol_table() -> pynini.SymbolTable | None:
    if not os.path.exists(_SYMS_PATH):
        return None
    try:
        return pynini.SymbolTable.read(_SYMS_PATH)
    except Exception:
        logger.warning(f"Failed to load symbol table from {_SYMS_PATH}")
        return None


def save_fst(kind: str, name: str, fst_kind: str, fst: pynini.Fst) -> None:
    path = _fst_path(kind, name, fst_kind)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fst.write(path)


def load_fst(kind: str, name: str, fst_kind: str) -> pynini.Fst | None:
    path = _fst_path(kind, name, fst_kind)
    if not os.path.exists(path):
        return None
    try:
        return pynini.Fst.read(path)
    except Exception:
        logger.warning(f"Failed to load FST from {path}")
        return None


def max_directory_mtime(directory: str):
    yaml_glob = glob(os.path.join(directory, "*.yaml"))
    csv_glob = glob(os.path.join(directory, "*.csv"))
    return max(os.path.getmtime(f) for f in yaml_glob + csv_glob + [directory])


def get_hashable_args_and_kwargs(args, kwargs):
    hashable_args = []
    for arg in args:
        hashable_arg = arg
        if type(arg) is list:
            hashable_arg = tuple(arg)
        elif type(arg) is dict:
            hashable_arg = frozendict(arg)
        try:
            hash(hashable_arg)
        except Exception as e:
            raise ValueError(f"Could not hash kwarg {value} with key {key}: {e}")

        hashable_args.append(hashable_arg)

    hashable_kwargs = {}
    for key, value in kwargs.items():
        hashable_value = value
        if type(value) is list:
            hashable_value = tuple(value)
        elif type(value) is dict:
            hashable_value = frozendict(value)
        try:
            hash(hashable_value)
        except Exception as e:
            raise ValueError(f"Could not hash kwarg {value} with key {key}: {e}")

        hashable_kwargs[key] = hashable_value

    return hashable_args, hashable_kwargs


def observed_cache(directories: list[str]):
    """
    Decorator to invalidate an entire function cache when file
    inside the specified directories changes.
    """

    directory_mtimes = {
        directory: max_directory_mtime(directory) for directory in directories
    }

    def decorator(func):
        cached_func = lru_cache(maxsize=128)(func)

        @wraps(func)
        def wrapper(*args, **kwargs):

            clear_cache = False
            for directory, mtime in directory_mtimes.items():
                new_mtime = max_directory_mtime(directory)
                if new_mtime > mtime:
                    directory_mtimes[directory] = new_mtime
                    clear_cache = True
            if clear_cache:
                logger.info(
                    f"Invalidated cache for {func.__name__}, rebuilding output..."
                )
                cached_func.cache_clear()

            try:
                args, kwargs = get_hashable_args_and_kwargs(args, kwargs)
            except Exception as e:
                logger.exception(
                    f"Error hashing args, building function output without caching. {e}"
                )
                return func(*args, **kwargs)

            return cached_func(*args, **kwargs)

        # Expose cache operations upstream (allows manual clearing if needed)
        wrapper.cache_clear = cached_func.cache_clear
        wrapper.cache_info = cached_func.cache_info
        return wrapper

    return decorator
