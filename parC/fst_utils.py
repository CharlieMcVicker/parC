import pynini
from dataclasses import dataclass, field
from loguru import logger
from frozendict import frozendict


class ReservedSymbolMixin:
    """
    Mixin class for registries to define reserved symbols that cannot be used as
    inventory item values. This is to prevent collisions between user-defined
    inventory items and special symbols used in pattern/rule contexts.
    """

    bow = "[BOW]"
    eow = "[EOW]"
    insert = "[INSERT]"
    substitute = "[SUBSTITUTE]"
    delete = "[DELETE]"

    word_edge = "#"
    phone_ref = "<Phone>"
    flag_ref = "<tag>"
    sigma_ref = "<Sigma>"
    dot = "."
    epsilon_ref = "<Empty>"
    boundary_ref = "<Boundary>"

    affix_boundary = "-"
    clitic_boundary = "="
    periphrasis_break = "_"

    star = "*"
    plus = "+"
    optional = "?"
    union = "|"
    caret = "^"
    left_paren = "("
    right_paren = ")"
    # curly braces indicate union of tokens, e.g. {A B} matches either A or B
    # similar to square brackets in regex
    left_brace = "{"
    right_brace = "}"

    left_delimiters = (left_paren, left_brace)
    right_delimiters = (right_paren, right_brace)
    unary_operators = (star, plus, optional)
    pipe_operator = union  # (for now) pipe operator is only binary operator
    caret_operator = caret  # for negation in braced expressions
    reserved_refs = (phone_ref, flag_ref, epsilon_ref,
                     dot, sigma_ref, boundary_ref)
    bow_eow_tags = (bow, eow)
    edit_tags = (insert, substitute, delete)
    boundary_symbols = (affix_boundary, clitic_boundary, periphrasis_break)

    reserved_symbols = (
        left_delimiters
        + right_delimiters
        + unary_operators
        + (pipe_operator, caret_operator)
        + reserved_refs
        + bow_eow_tags
        + edit_tags
        + boundary_symbols
        + (word_edge,)
    )


def is_acceptor(fsa: pynini.Fst) -> bool:
    if not isinstance(fsa, pynini.Fst):
        raise ValueError(f"Expected pynini.Fst but got {fsa} for fsa arg.")
    return fsa.properties(pynini.ACCEPTOR, True)


def stringify_features(feature_values: set[tuple[str, str]] | dict[str, str]) -> str:
    if isinstance(feature_values, (dict, frozendict)):
        feature_values = list(feature_values.items())
    return "".join(f"[{f}={v}]" for f, v in sorted(feature_values))
