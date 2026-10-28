"""Label horizons vary; unknown/censored labels never become negative examples."""
from pathlib import Path
import numpy as np
import pandas as pd
from common import DAY, require
from history_store import META, batches, training_batches, feature_names


def container_available():
    """Bytes this process tree may still allocate: host available, capped by the cgroup v2 limit.

    Containers report host RAM in psutil/free; the cgroup limit is what actually triggers SIGKILL.
    """
    import psutil
    value = psutil.virtual_memory().available
    try:
        root = Path('/sys/fs/cgroup'); limit = (root/'memory.max').read_text().strip()
        if limit != 'max':
            stat = dict(line.split() for line in (root/'memory.stat').read_text().splitlines())
            used = int((root/'memory.current').read_text()) - int(stat.get('inactive_file', 0))
            value = min(value, int(limit) - used)
    except (FileNotFoundError, ValueError):
        pass
    return value


def targets(meta, horizon, boundary, lead=900):
    t = meta["time"].to_numpy(np.int64)
    ue = meta["failure"].to_numpy(np.int64)
    end = np.minimum(meta["observed_end"].to_numpy(np.int64), boundary-1)
    delta = ue-t
    # Outcome-independent maturity filter keeps the cohort selection comparable.
    target_end=t+lead+horizon*DAY
    y = ((ue>=0) & (delta>=lead) & (ue<=target_end) & (ue<=end)).astype(np.uint8)
    # Keep the split boundary outcome-independent; terminal failures establish their own label.
    valid = (meta['observed_start'].to_numpy() <= t) & (target_end < boundary)
    valid &= ((end>=target_end) | (y==1)) & ((ue<0) | (delta>=lead))
    return valid, y


def daily_meta(cache, split, manifest, lead):
    from evaluation import period_cohort, annotate
    frames = [b.to_pandas() for b in batches(cache, "daily_"+split, META)]
    require(frames, f"No daily {split} data")
    m = pd.concat(frames, ignore_index=True)
    registry = pd.read_parquet(Path(cache)/"device_metadata.parquet")
    cohort = period_cohort(registry, manifest["windows"][split])
    return annotate(m,cohort,lead), cohort


def build_training(c, recipe, manifest, folder, lead, memory_gb, device_mask=None, return_devices=False):
    """Two passes into a disk memmap; every eligible row or daily history is retained.

    device_mask (bool per device code) keeps only those devices' rows, for grouped
    cross-validation. The audited sample identity is still verified on the full TRAIN
    set and then restricted, so a fold reuses exactly the sealed rows of its devices.
    """
    import psutil
    horizon = recipe["label_horizon_days"]
    features = feature_names(manifest["features"], recipe["feature_set"])
    lo, hi = manifest["windows"]["train"]
    counts = np.zeros(manifest["included_devices"], np.int64)
    positive_devices = set()
    sample_hash = __import__("hashlib").sha256()
    candidate_samples = no_history = 0
    total = pos = 0
    row_devices = []
    for b in training_batches(c, recipe, manifest, meta_only=True):
        m = b.to_pandas(); use, y = targets(m, horizon, hi, lead)
        use &= (m.time>=lo).to_numpy()
        candidate_samples += int(((m.time>=lo)&(m.time<hi)).sum())
        no_history += int((use & (m.history_rows<=0).to_numpy()).sum())
        use &= (m.history_rows>0).to_numpy()
        sample_hash.update(m.loc[use,["device","time"]].to_numpy(dtype="<i8").tobytes())
        np.add.at(counts, m.device.to_numpy()[use], 1)
        positive_devices.update(m.device.to_numpy()[use & (y==1)].tolist())
        if device_mask is not None:row_devices.append(m.device.to_numpy()[use].astype(np.int64))
        total += int(use.sum()); pos += int(y[use].sum())
    require(total>pos>0, "Training combination has no positive or negative labels")
    from training_selector import selection, slice_indices
    indices, selected_weights, sampling_audit = selection(c,recipe,total,pos,sample_hash.hexdigest())
    full_total = total
    if device_mask is not None:
        mask = np.asarray(device_mask, bool)
        require(len(mask)==manifest["included_devices"], "Device mask length differs from registry")
        keep = mask[np.concatenate(row_devices)] if row_devices else np.zeros(0, bool)
        del row_devices
        if indices is None:
            indices = np.flatnonzero(keep).astype(np.int64)
        else:
            chosen = keep[np.asarray(indices)]
            indices = np.asarray(indices)[chosen]; selected_weights = np.asarray(selected_weights)[chosen]
        sampling_audit = dict(sampling_audit or {}, cross_validation_devices=int(mask.sum()),
                              cross_validation_rows=int(len(indices)))
        positive_devices = {d for d in positive_devices if mask[d]}
        counts = np.where(mask, counts, 0)
    matrix_total = len(indices) if indices is not None else total
    matrix_bytes = matrix_total*len(features)*4
    estimate = matrix_bytes*2 + total*40 + max(256*1024**2, min(2*1024**3, matrix_bytes))
    available = min(memory_gb*1024**3, container_available()*.85)
    require(estimate<available, f"Estimated matrix RAM {estimate/1024**3:.1f} GiB exceeds {available/1024**3:.1f}; raise memory budget/instance RAM. No negative sampling performed.")
    path = Path(folder)/"train_matrix.npy"
    import shutil
    require(shutil.disk_usage(folder).free>matrix_bytes+256*1024**2,
            f"Need at least {(matrix_bytes+256*1024**2)/1024**3:.1f} GiB free scratch disk")
    x = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(matrix_total,len(features)))
    yout = np.empty(matrix_total, np.uint8); w = np.empty(matrix_total, np.float32)
    devices = np.empty(matrix_total, np.int64) if return_devices else None
    offset = 0; full_offset = 0
    for b in training_batches(c, recipe, manifest):
        m = b.to_pandas(); use, y = targets(m, horizon, hi, lead)
        use &= (m.time>=lo).to_numpy() & (m.history_rows>0).to_numpy()
        n = int(use.sum())
        chosen, sl = slice_indices(indices,full_offset,n)
        frame = m.loc[use]; labels = y[use]
        if chosen is not None:
            frame=frame.iloc[chosen];labels=labels[chosen]
        take=len(frame)
        x[offset:offset+take] = frame[features].to_numpy(np.float32)
        yout[offset:offset+take] = labels
        codes = frame.device.to_numpy()
        if devices is not None:devices[offset:offset+take] = codes
        if chosen is not None and selected_weights is not None:w[offset:offset+take] = selected_weights[sl]
        else:w[offset:offset+take] = 1 if recipe['base_weight']=='sample' else 1/counts[codes]
        offset += take; full_offset += n
    require(full_offset==full_total and offset==matrix_total,'Training row count drift')
    if device_mask is not None:
        pos = int(yout.sum())
        require(matrix_total>pos>0, 'Cross-validation fold has no positive or negative labels')
    elif indices is not None:
        require(int(yout.sum())==pos,'Selective sample lost positive rows')
        require(np.isclose(w[yout==0].sum(dtype=np.float64),full_total-pos,rtol=1e-6),'Negative loss mass changed')
    x.flush()
    ratio = float(w[yout==0].sum(dtype=np.float64)/w[yout==1].sum(dtype=np.float64))
    positive_weight = {"ratio":ratio, "sqrt_ratio":ratio**.5, "unit":1.}[recipe["weight_policy"]]
    audit = dict(
        sampling=sampling_audit, training_samples=matrix_total, positive_samples=pos, sample_membership_sha256=sample_hash.hexdigest(), training_devices=int((counts>0).sum()),
        positive_devices=len(positive_devices),candidate_samples_in_train_interval=candidate_samples,
        eligible_no_history_samples=no_history,
        excluded_maturity_or_lead_samples=candidate_samples-total-no_history,
        weighted_class_ratio=ratio, positive_weight=positive_weight,
        estimated_matrix_ram_gib=estimate/1024**3,
        weight_definition="each sample = 1" if recipe["base_weight"]=="sample" else "each device total base weight = 1")
    if return_devices:
        return x, yout, w, features, positive_weight, audit, devices
    return x, yout, w, features, positive_weight, audit
