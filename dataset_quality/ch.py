"""Caliński–Harabasz variance ratio on fixed image features and current labels."""
from __future__ import annotations

import numpy as np


def score(features, labels) -> dict:
    """Return BGSS/(k-1) divided by WGSS/(n-k), without fitting clusters."""
    values = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if (values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all()
            or labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in 'iu'):
        raise ValueError('Use finite [N,D] features and aligned integer [N] labels.')
    classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    n, k = len(values), len(classes)
    if not 2 <= k < n:
        return {'metric': 'calinski_harabasz', 'score': None, 'status': 'invalid_partition',
                'higher_is_better': True, 'sample_count': n, 'num_classes': k,
                'class_labels': classes.tolist(), 'class_counts': counts.tolist(),
                'reason': 'CH requires 2 <= number of nonempty classes < sample count.',
                'ground_truth_verified': False}
    centers = np.zeros((k, values.shape[1]), dtype=np.float64)
    np.add.at(centers, inverse, values)
    centers /= counts[:, None]
    within = float(np.square(values - centers[inverse]).sum())
    between = float((counts[:, None] * np.square(centers - values.mean(axis=0))).sum())
    if not np.isfinite(within) or not np.isfinite(between):
        raise ValueError('Feature sums of squares overflow; rescale the fixed features before evaluation.')
    result = {'metric': 'calinski_harabasz', 'score': None, 'status': 'ok',
              'higher_is_better': True, 'sample_count': n, 'num_classes': k,
              'class_labels': classes.tolist(), 'class_counts': counts.tolist(),
              'within_sum_squares': within, 'between_sum_squares': between,
              'formula': '(BGSS / (k - 1)) / (WGSS / (n - k))',
              'ground_truth_verified': False}
    if within == 0:
        result.update(status='zero_within_dispersion',
                      reason='The variance ratio is undefined at WGSS=0; no finite score is reported.',
                      mathematical_limit='positive_infinity' if between > 0 else 'undefined_0_over_0')
    else:
        value = float((between / (k - 1)) / (within / (n - k)))
        if np.isfinite(value):
            result['score'] = value
        else:
            result.update(status='nonfinite_ratio',
                          reason='The variance ratio exceeds floating-point range; no finite score is reported.')
    return result
