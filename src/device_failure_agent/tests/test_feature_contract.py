"""Regression: sparse first devices must not invalidate a useful feature."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from common import save,sha
from feature_programs import training_samples,register,apply,validate_source

SOURCE="def transform(frame):\n    x = frame['smart_5_raw'].rolling('7D', min_periods=1).mean()\n    return pd.DataFrame({'fe_mean': x}, index=frame.index)"

class FeatureRegression(unittest.TestCase):
    def test_missing_first_six_and_late_training_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'cache';(cache/'rows').mkdir(parents=True)
            rows=[]
            for device in range(20):
                for step in range(8):
                    rows.append(dict(device=device,time=step*86400,
                        smart_5_raw=float(device) if device>=6 and step<6 else np.nan,
                        failure=-1,observed_start=0,observed_end=7*86400,history_rows=step+1))
            # A validation-only field value must never enter profile/validation samples.
            rows[-1]['smart_5_raw']=999999.
            frame=pd.DataFrame(rows)
            for i,indices in enumerate(np.array_split(np.arange(len(frame)),4)):
                piece=frame.iloc[indices]
                piece.to_parquet(cache/'rows'/f'part-{i:05d}.parquet',index=False)
            m=dict(features=['smart_5_raw'],windows={'train':[0,6*86400]},
                files={p.relative_to(cache).as_posix():sha(p) for p in cache.rglob('*.parquet')})
            save(cache/'manifest.json',m)
            c=dict(output_root=tmp,max_features_per_program=8,max_total_features=200)
            samples=training_samples(c,'base',m)
            self.assertTrue(any(np.isfinite(x).any() for t,x,n in samples),'First-six bias still present')
            for t,x,n in samples:
                self.assertLess(t.max(),6*86400);self.assertFalse((x==999999).any())
            answer=dict(source=SOURCE,features=['fe_mean'],reason='Trailing SMART level')
            key=register(c,answer,'base',samples)
            self.assertEqual(len(key),64)
            again=training_samples(c,'base',m)
            for a,b in zip(samples,again):
                np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])

    def test_gap_helper_and_declared_examples(self):
        from feature_contract import contract
        schema=contract()
        for example in schema['examples']:
            times=np.array([0,60,240,300]);x=np.array([[1.],[np.nan],[3.],[4.]])
            full=apply(example['source'],times,x,['metric'],example['features'])
            for cut in (1,2,3):
                np.testing.assert_allclose(full[:cut],apply(example['source'],times[:cut],x[:cut],['metric'],example['features']),equal_nan=True)

    def test_actionable_rejection(self):
        with self.assertRaisesRegex(ValueError,'to_series'):
            validate_source("def transform(frame):\n    t = frame.index.to_series()\n    return pd.DataFrame({'fe_t': t}, index=frame.index)")
        with self.assertRaises(ValueError):
            validate_source("def transform(frame):\n    return pd.DataFrame({'fe_x': frame['metric'].shift(-1)}, index=frame.index)")

if __name__=='__main__':unittest.main()
