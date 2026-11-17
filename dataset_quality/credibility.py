"""Data Credibility with the paper's first/second/third-order HOC estimator.

The reference features must come from one fixed encoder. Each anchor and its
two nearest other images form a triplet. The estimator assumes that the three
images share a latent true class and that their observed labels are independent
conditional on that class, with a common row-stochastic transition matrix T.
Appendix B minimizes the sum of three (unsquared) L2 consensus residuals.
We store ordered label tuples instead of cyclic-shift coordinates: this merely
permutes the entries of each consensus vector and leaves its L2 norm unchanged.

Latent row names are not identifiable without clean labels. We align rows to
observed class names by maximum diagonal mass, which assumes class names are
mostly preserved. A global semantic label permutation cannot be detected by
this diagnostic. These implementation choices are reported, not evidence of
verified semantic label accuracy.
"""
from __future__ import annotations

from math import isfinite, sqrt

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
import torch
from torch.nn import functional as F
from tqdm import tqdm

from shared.precision import full_precision


def nearest_neighbor_triplets(features, chunk_size: int = 256, device="cpu", *, progress: bool = True):
    """Return CPU int64 [N,3] (anchor, nearest, second-nearest), excluding self.

    Only a chunk-by-N cosine matrix is retained. Exact cosine ties use the
    smaller source index, so duplicate features cannot select the anchor or
    change tie order when the query chunk size changes.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    values = torch.as_tensor(features).detach()
    if (values.ndim != 2 or len(values) < 3 or values.size(1) < 1
            or not values.is_floating_point() or not torch.isfinite(values).all()):
        raise ValueError("Features must be finite floating-point [N,D], with N>=3 and D>=1.")
    device = torch.device(device)
    values = values.to(device=device, dtype=torch.float32)
    norms = values.norm(dim=1)
    if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(norms).all()):
        raise ValueError("Features must remain finite in FP32 cosine arithmetic.")
    if not bool(norms.gt(0).all()):
        raise ValueError("Zero feature vectors do not define cosine neighbors.")
    result = torch.empty((len(values), 3), dtype=torch.int64)
    result[:, 0] = torch.arange(len(values))
    old_tf32 = torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else None
    if old_tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad(), full_precision(device):
            values = F.normalize(values, dim=1)
            for start in tqdm(range(0, len(values), chunk_size), desc="Find neighbors",
                              unit="batch", mininterval=5, disable=not progress):
                end = min(start + chunk_size, len(values))
                similarities = values[start:end] @ values.T
                local = torch.arange(end - start, device=device)
                similarities[local, local + start] = -torch.inf
                scores, indices = similarities.topk(2, dim=1, largest=True, sorted=True)
                ties = (similarities == scores[:, 1, None]).sum(dim=1) > 1
                for row in torch.where(ties)[0].tolist():
                    candidates = torch.where(similarities[row] == scores[row, 1])[0]
                    if scores[row, 0] == scores[row, 1]:
                        indices[row] = candidates[:2]
                    else:
                        indices[row, 1] = candidates[0]
                result[start:end, 1:] = indices.cpu()
    finally:
        if old_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return result


def credibility_from_transition(transition) -> float:
    """Compute exactly 1 - ||T-I||_F / sqrt(2K) for a named-class matrix."""
    matrix = np.asarray(transition, dtype=np.float64)
    if (matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[0] != matrix.shape[1]
            or not np.isfinite(matrix).all() or (matrix < 0).any()
            or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8, rtol=1e-8)):
        raise ValueError("Transition matrix must be finite, nonnegative, square and row-stochastic.")
    score = 1.0 - float(np.linalg.norm(matrix - np.eye(len(matrix)), ord="fro")) / sqrt(2 * len(matrix))
    return float(np.clip(score, 0.0, 1.0))


def _integer_array(value, name, dimensions):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim != dimensions or value.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be an integer array with {dimensions} dimension(s).")
    return value.astype(np.int64, copy=False)


def _triplet_patterns(values):
    a, b, c = values.T
    return {"AAA": int(((a == b) & (b == c)).sum()),
            "AAB": int(((a == b) & (a != c)).sum()),
            "ABA": int(((a == c) & (a != b)).sum()),
            "ABB": int(((b == c) & (a != b)).sum()),
            "ABC": int(((a != b) & (a != c) & (b != c)).sum())}


def _unpack(parameters, classes):
    prior_logits = np.r_[parameters[:classes - 1], 0.0]
    row_logits = np.concatenate((parameters[classes - 1:].reshape(classes, classes - 1),
                                np.zeros((classes, 1))), axis=1)
    prior = np.exp(prior_logits - prior_logits.max())
    prior /= prior.sum()
    transition = np.exp(row_logits - row_logits.max(axis=1, keepdims=True))
    transition /= transition.sum(axis=1, keepdims=True)
    return prior, transition


def _objective(parameters, frequencies, classes, squared_initialization=False):
    prior, transition = _unpack(parameters, classes)
    empirical = (frequencies.sum(axis=(1, 2)), frequencies.sum(axis=2), frequencies)
    fitted = (prior @ transition,
              np.einsum('i,ia,ib->ab', prior, transition, transition, optimize=True),
              np.einsum('i,ia,ib,ic->abc', prior, transition, transition, transition, optimize=True))
    residuals = [model - data for model, data in zip(fitted, empirical)]
    norms = [float(np.linalg.norm(residual.ravel())) for residual in residuals]
    d1, d2, d3 = [2 * residual if squared_initialization else
                  (residual / norm if norm > 1e-16 else np.zeros_like(residual))
                  for residual, norm in zip(residuals, norms)]
    prior_gradient = (np.einsum('a,ia->i', d1, transition)
                      + np.einsum('ab,ia,ib->i', d2, transition, transition, optimize=True)
                      + np.einsum('abc,ia,ib,ic->i', d3, transition, transition, transition, optimize=True))
    transition_gradient = prior[:, None] * (
        d1[None, :] + np.einsum('jb,ib->ij', d2, transition)
        + np.einsum('aj,ia->ij', d2, transition)
        + np.einsum('jbc,ib,ic->ij', d3, transition, transition, optimize=True)
        + np.einsum('ajc,ia,ic->ij', d3, transition, transition, optimize=True)
        + np.einsum('abj,ia,ib->ij', d3, transition, transition, optimize=True))
    prior_logits_gradient = prior * (prior_gradient - prior @ prior_gradient)
    row_logits_gradient = transition * (
        transition_gradient - (transition * transition_gradient).sum(axis=1, keepdims=True))
    gradient = np.r_[prior_logits_gradient[:-1], row_logits_gradient[:, :-1].ravel()]
    return sum(norm * norm for norm in norms) if squared_initialization else sum(norms), gradient


def _align(prior, transition):
    rows, columns = linear_sum_assignment(-transition)
    order = np.empty(len(prior), dtype=np.int64)
    order[columns] = rows
    return prior[order], transition[order], order


def estimate_credibility(labels, triplets, num_classes: int, *, seed: int = 0,
                         restarts: int = 5, steps: int = 1000) -> dict:
    """Fit Appendix B's HOC consensus equations and report Data Credibility.

    The same ``triplets`` must be reused for every before/after label candidate.
    Scores are null when classes are absent, the fit fails to converge, or the
    fitted latent mixture is numerically degenerate. Residuals and all restart
    outcomes are retained even when the score is unavailable.
    """
    if (type(num_classes) is not int or num_classes < 2 or type(seed) is not int or seed < 0
            or type(restarts) is not int or restarts < 1 or type(steps) is not int or steps < 1):
        raise ValueError("Use K>=2, seed>=0, and positive integer restarts/steps.")
    labels = _integer_array(labels, "labels", 1)
    triplets = _integer_array(triplets, "triplets", 2)
    if len(labels) < 3 or (labels < 0).any() or (labels >= num_classes).any():
        raise ValueError("Labels must contain at least three class indices in [0,K).")
    if (triplets.shape[1] != 3 or not len(triplets) or (triplets < 0).any()
            or (triplets >= len(labels)).any()
            or (triplets[:, 0] == triplets[:, 1]).any()
            or (triplets[:, 0] == triplets[:, 2]).any()
            or (triplets[:, 1] == triplets[:, 2]).any()):
        raise ValueError("Triplets must be nonempty [M,3] valid, distinct source indices.")
    observed = labels[triplets]
    counts = np.bincount((observed[:, 0] * num_classes + observed[:, 1]) * num_classes
                         + observed[:, 2], minlength=num_classes ** 3).reshape((num_classes,) * 3)
    frequencies = counts.astype(np.float64) / len(triplets)
    marginal_counts = np.stack([np.bincount(observed[:, position], minlength=num_classes)
                                for position in range(3)])
    result = {
        "score": None, "status": "pending", "metric": "data_credibility",
        "formula": "1 - frobenius_norm(T - I) / sqrt(2*K)",
        "transition_axes": {"row": "estimated_latent_true_class", "column": "observed_candidate_label"},
        "sample_count": len(labels), "triplet_count": len(triplets), "num_classes": num_classes,
        "observed_class_counts": np.bincount(labels, minlength=num_classes).tolist(),
        "triplet_patterns": _triplet_patterns(observed),
        "triplet_joint_counts": counts.tolist(), "position_label_counts": marginal_counts.tolist(),
        "estimator": "HOC_first_second_third_order_consensus_L2",
        "objective": "sum_nu=1..3 ||empirical_consensus_nu - model_consensus_nu||_2",
        "consensus_coordinates": "ordered_label_tuples_permutation_of_paper_cyclic_shift_coordinates",
        "optimization": "softmax_parameters_L-BFGS-B_multiple_restarts_logit_bound_20",
        "initialization": "squared_consensus_residual_warm_start_then_paper_unsquared_L2_objective",
        "row_alignment": "maximum_diagonal_mass_Hungarian",
        "assumptions": ["All three neighbor images share one latent true class.",
                        "Their observed label noises are conditionally independent with one common T.",
                        "Latent class names are aligned by maximum diagonal mass; semantic permutations are unidentifiable.",
                        "Initialization and alignment favor preserved class names, without using ground truth."],
        "ground_truth_verified": False, "transition_matrix": None, "class_prior": None,
    }
    if (marginal_counts.sum(axis=0) == 0).any():
        result.update(status="insufficient_observed_classes",
                      reason="At least one configured observed class is absent; the K-class noise model is not identifiable.")
        return result
    rng = np.random.default_rng(seed)
    initial_prior = marginal_counts.mean(axis=0) / len(triplets)
    fits = []
    for restart in range(restarts):
        random_rows = rng.dirichlet(np.ones(num_classes), size=num_classes)
        strength = 0.9 if restart == 0 else (0.0 if restart == restarts - 1 else rng.uniform(0.5, 0.95))
        initial_transition = strength * np.eye(num_classes) + (1 - strength) * random_rows
        prior = initial_prior if restart == 0 else rng.dirichlet(1 + 20 * initial_prior)
        initial = np.r_[np.log(prior[:-1] / prior[-1]),
                        np.log(initial_transition[:, :-1] / initial_transition[:, -1, None]).ravel()]
        bounds = [(-20.0, 20.0)] * len(initial)
        # A smooth moment fit only initializes the solver. The reported loss,
        # restart selection and final estimate all use Appendix B's unsquared
        # norms; this avoids stopping at a kink of the first-order residual.
        warm = minimize(_objective, initial, args=(frequencies, num_classes, True),
                        method="L-BFGS-B", jac=True, bounds=bounds,
                        options={"maxiter": min(steps, 200), "ftol": 1e-12, "gtol": 1e-8})
        fit = minimize(_objective, warm.x, args=(frequencies, num_classes), method="L-BFGS-B", jac=True,
                       bounds=bounds,
                       options={"maxiter": steps, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40})
        prior, transition, order = _align(*_unpack(fit.x, num_classes))
        fits.append((fit, prior, transition, order))
    finite = [index for index, (fit, *_) in enumerate(fits) if isfinite(float(fit.fun))]
    if not finite:
        result.update(status="nonfinite_fit", reason="No finite HOC objective was obtained.")
        return result
    best_index = min(finite, key=lambda index: fits[index][0].fun)
    converged = [index for index in finite if fits[index][0].success
                 and fits[index][0].fun <= fits[best_index][0].fun + 1e-7]
    if converged:
        best_index = min(converged, key=lambda index: fits[index][0].fun)
    best, prior, transition, order = fits[best_index]
    fitted = np.einsum("i,ia,ib,ic->abc", prior, transition, transition, transition, optimize=True)
    singular_values = np.linalg.svd(transition, compute_uv=False)
    estimate = credibility_from_transition(transition)
    position_frequencies = marginal_counts / len(triplets)
    consensus_errors = [float(np.linalg.norm((prior @ transition) - frequencies.sum(axis=(1, 2)))),
                        float(np.linalg.norm(np.einsum('i,ia,ib->ab', prior, transition, transition)
                                             - frequencies.sum(axis=2))),
                        float(np.linalg.norm(fitted - frequencies))]
    result.update(transition_matrix=transition.tolist(), class_prior=prior.tolist(),
                  latent_row_for_observed_class=order.tolist(),
                  fit={"converged": bool(best.success), "iterations": int(best.nit),
                       "objective": float(best.fun),
                       "consensus_order_L2_residuals": consensus_errors,
                       "total_variation_residual": float(np.abs(fitted - frequencies).sum() / 2),
                       "frobenius_residual": float(np.linalg.norm(fitted - frequencies)),
                       "max_position_marginal_tv": float(max(np.abs(position_frequencies[a]
                           - position_frequencies[b]).sum() / 2 for a in range(3) for b in range(a + 1, 3))),
                       "minimum_transition_singular_value": float(singular_values[-1]),
                       "best_restart": best_index, "seed": seed,
                       "restarts": [{"index": index, "converged": bool(fit.success),
                                     "iterations": int(fit.nit), "objective": float(fit.fun),
                                     "aligned_score_estimate": credibility_from_transition(matrix),
                                     "message": str(fit.message)}
                                    for index, (fit, _, matrix, _) in enumerate(fits)]})
    if not best.success:
        result.update(status="optimizer_not_converged", reason=str(best.message))
    elif prior.min() < 1e-5 or singular_values[-1] < 1e-5:
        result.update(status="degenerate_latent_mixture",
                      reason="A fitted latent class vanished or T is numerically rank deficient.")
    else:
        result.update(status="ok", score=estimate)
    return result
