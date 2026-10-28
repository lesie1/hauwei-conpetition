from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from common import DAY,read,save,sha,require
STATS=("mean","max","std","last","missing_fraction","slope_per_day")
def feature_names(base):
    return [f"{x}__{s}" for s in STATS for x in base] + [
        "log1p_history_rows", "history_span_days", "last_row_age_days"]


class Writer:
    def __init__(self, path):
        self.path = Path(path); self.path.mkdir(parents=True, exist_ok=True)
        self.frames = []; self.n = 0; self.part = 0; self.buffer_bytes = 0
    def add(self, frame):
        if not len(frame):
            return
        self.frames.append(frame); self.n += len(frame)
        # Frames are owned by the writer after add; count each frame only once.
        self.buffer_bytes += int(frame.memory_usage(deep=True).sum())
        if self.n >= 200000 or self.buffer_bytes>=128*1024**2:
            self.flush()
    def flush(self):
        if self.frames:
            pq.write_table(pa.Table.from_pandas(pd.concat(self.frames, ignore_index=True),
                                               preserve_index=False),
                           self.path/f"part-{self.part:05d}.parquet", compression="zstd",
                           row_group_size=65536)
            self.part += 1; self.frames = []; self.n = 0; self.buffer_bytes = 0


def cumulative_features(times, x, anchors):
    """One forward pass. Future rows cannot affect any earlier anchor."""
    width = x.shape[1]
    count = np.zeros(width, np.int64)
    mean = np.zeros(width); m2 = np.zeros(width)
    time_mean = np.zeros(width); time_m2 = np.zeros(width); cross = np.zeros(width)
    high = np.full(width, -np.inf)
    end = 0
    result = np.full((len(anchors), width*len(STATS)+3), np.nan, np.float32)
    lengths = np.zeros(len(anchors), np.int64)
    for j, anchor in enumerate(anchors):
        stop = np.searchsorted(times, anchor, side="right")
        a = np.asarray(x[end:stop], np.float64)
        if len(a):
            valid = np.isfinite(a); n = valid.sum(0)
            avg = np.divide(np.where(valid, a, 0).sum(0), n, out=np.zeros(width), where=n>0)
            q = np.where(valid, a-avg, 0)
            delta = avg-mean; newcount = count+n
            td = (times[end:stop]-times[0]).astype(np.float64)[:,None]/DAY
            tavg = np.divide(np.where(valid,td,0).sum(0),n,out=np.zeros(width),where=n>0)
            tq = np.where(valid,td-tavg,0)
            tdelta = tavg-time_mean
            merge_weight = np.divide(count*n,newcount,out=np.zeros(width),where=newcount>0)
            cross += (q*tq).sum(0)+delta*tdelta*merge_weight
            time_m2 += (tq*tq).sum(0)+tdelta*tdelta*merge_weight
            time_mean += np.divide(tdelta*n,newcount,out=np.zeros(width),where=newcount>0)
            m2 += (q*q).sum(0) + np.divide(delta*delta*count*n, newcount,
                                          out=np.zeros(width), where=newcount>0)
            mean += np.divide(delta*n, newcount, out=np.zeros(width), where=newcount>0)
            count = newcount
            high = np.maximum(high, np.where(valid, a, -np.inf).max(0))
        end = stop; lengths[j] = stop
        if stop:
            avg = mean.copy(); avg[count==0] = np.nan
            maximum = high.copy(); maximum[count==0] = np.nan
            std = np.sqrt(np.divide(m2, count, out=np.zeros(width), where=count>0))
            std[count==0] = np.nan
            slope = np.divide(cross,time_m2,out=np.full(width,np.nan),where=time_m2>1e-15)
            result[j] = np.r_[avg, maximum, std, x[stop-1], 1-count/stop, slope,
                np.log1p(stop), (times[stop-1]-times[0])/DAY, (anchor-times[stop-1])/DAY]
    return result, lengths


def verify(cache):
    cache = Path(cache); m = read(cache/"manifest.json")
    actual = {p.relative_to(cache).as_posix() for p in cache.rglob("*.parquet")}
    require(actual == {x for x in m["files"] if x.endswith(".parquet")}, "Cache shard set changed")
    for f, h in m["files"].items():
        require(sha(cache/f) == h, f"Cache modified: {f}")
    return m


def batches(cache, name, columns=None):
    for p in sorted((Path(cache)/name).glob("*.parquet")):
        yield from pq.ParquetFile(p).iter_batches(batch_size=65536, columns=columns)
