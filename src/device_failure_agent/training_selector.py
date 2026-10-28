"""Reuse the audited TRAIN-only selective sample across parameter/feature controls."""
from pathlib import Path
import numpy as np
from common import read, sha, require


def selection(c, recipe, total, positives, membership):
    ref = c.get('sampling_reference')
    if not ref:
        return None, None, {}
    require(recipe['sample_unit'] == 'event' and recipe['base_weight'] == 'sample'
            and recipe['label_horizon_days'] == 7, 'Frozen selective cohort requires original event/sample/7-day labels')
    seal = read(ref)
    for name, expected in seal['files'].items():
        require(sha(name) == expected, 'Frozen sampling input changed: '+name)
    require(total == seal['full_training_samples'] and positives == seal['positive_samples']
            and membership == seal['full_membership_sha256'], 'TRAIN row identity changed; do not reuse row positions')
    ix = np.load(seal['row_ids'], mmap_mode='r', allow_pickle=False)
    weights = np.load(seal['weights'], mmap_mode='r', allow_pickle=False)
    require(ix.ndim == weights.ndim == 1 and len(ix) == len(weights) > 0, 'Invalid selection arrays')
    require(ix[0] >= 0 and ix[-1] < total and np.all(np.diff(ix)>0), 'Selected IDs must be unique sorted TRAIN indices')
    require(np.isfinite(weights).all() and (weights>0).all(), 'Invalid sampling weights')
    return ix, weights, dict(strategy=seal.get('strategy','frozen_selective_device_week_10pct_20pct_hard'),
        selection_seal_sha256=sha(ref), full_training_samples=total,
        full_positive_samples=positives, full_membership_sha256=membership,
        selection_membership_sha256=seal['selected_membership_sha256'],
        all_positives_retained=True, selection_uses_validation_or_test=False,
        note='Reuse earlier TRAIN-only mature-prefix mining and fixed device/week quotas; no re-mining from validation errors.')


def slice_indices(indices, offset, count):
    if indices is None:
        return None, None
    a,b = np.searchsorted(indices,[offset,offset+count])
    return np.asarray(indices[a:b]-offset,dtype=np.int64), slice(a,b)
