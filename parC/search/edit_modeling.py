"""
Helper functions for modeling edit probabilities.
At present supports a unigram edit model using a transition matrix.
TODO: support n-gram edit models for wider context using `nltk.lm`.
"""

from scipy.stats import entropy, dirichlet
import numpy as np

"""
Transition matrix over edit pairs.
"""


def get_random_transition_matrix(alphabet: list[str], alpha: float) -> np.ndarray:
    """
    Generate a random transition matrix of size n x n using a Dirichlet distribution.
    """
    n = len(alphabet)
    mat = np.random.dirichlet([alpha] * n, size=n)

    # mask diagonal to prevent self-transitions (i.e., no edit)
    np.fill_diagonal(mat, 0)

    # mask any index which is None in alphabet
    for i, token in enumerate(alphabet):
        if token is None:
            mat[:, i] = 0
            mat[i, :] = 0

    probs = np.divide(
        mat, mat.sum(axis=1, keepdims=True), where=mat.sum(axis=1, keepdims=True) != 0
    )  # Ensure rows sum to 1

    return probs


def kl_divergence_from_uniform(probs: np.ndarray) -> float:
    """
    Calculate the KL divergence of a probability distribution from the uniform distribution.
    """
    uniform_dist = np.ones_like(probs) / len(probs)
    uniform_dist = uniform_dist / uniform_dist.sum()  # Ensure it's a valid distribution
    kl_div = entropy(probs, uniform_dist)
    return kl_div


def conditional_kl_divergence_from_uniform(
    transition_matrix: np.ndarray, prior_probs: np.ndarray
) -> float:
    """
    Calculate the average KL divergence of each row of the transition matrix
    from the uniform distribution weighted by the prior probabilities of the rows.
    """
    kl_divs = [
        prior * kl_divergence_from_uniform(row)
        for (prior, row) in zip(prior_probs, transition_matrix)
    ]
    return np.mean(kl_divs)
