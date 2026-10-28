import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import data_ingest
from common import read, save, sha, DAY
from history_math import Writer
from resume_normalization import prepare_checkpoint, finish_checkpoint
from settings import DEFAULTS
from tests.test_workflow import fixture


class LegacyWriter(Writer):
    def add(self,frame):
        if not len(frame):return
        self.frames.append(frame);self.n+=len(frame)
        if self.n>=200000 or sum(int(f.memory_usage(deep=True).sum()) for f in self.frames)>=128*1024**2:self.flush()


class WriterTests(unittest.TestCase):
    def test_memory_accounting_is_linear(self):
        original=pd.DataFrame.memory_usage;calls=[]
        def count(f,*a,**kw):calls.append(1);return original(f,*a,**kw)
        with tempfile.TemporaryDirectory() as tmp,patch.object(pd.DataFrame,'memory_usage',count):
            w=Writer(tmp)
            for i in range(30):w.add(pd.DataFrame({'x':np.arange(10,dtype=np.float32)}))
            self.assertEqual(len(calls),30)
            w.flush();self.assertEqual(w.buffer_bytes,0)

    def test_memory_flush_preserves_exact_parquet_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=LegacyWriter(Path(tmp)/'old');b=Writer(Path(tmp)/'new')
            frames=[pd.DataFrame({'x':[float(i),np.nan],'text':['a','b']}) for i in range(5)]
            with patch.object(pd.DataFrame,'memory_usage',return_value=pd.Series([80*1024**2])):
                for w in (a,b):
                    for f in frames:w.add(f.copy())
                    w.flush()
            old=sorted(a.path.glob('*.parquet'));new=sorted(b.path.glob('*.parquet'))
            self.assertEqual(len(old),3);self.assertEqual([sha(p) for p in old],[sha(p) for p in new])

    def test_row_flush_and_empty_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=LegacyWriter(Path(tmp)/'old');b=Writer(Path(tmp)/'new')
            for w in (a,b):
                w.add(pd.DataFrame({'x':[]}))
                for n in (100001,100001,3):w.add(pd.DataFrame({'x':np.arange(n,dtype=np.float32)}))
                w.flush()
            self.assertEqual([sha(p) for p in sorted(a.path.glob('*.parquet'))],
                             [sha(p) for p in sorted(b.path.glob('*.parquet'))])


class RecoveryTests(unittest.TestCase):
    def setup_case(self,root):
        mapping=fixture(root/'data','hdd')
        p=root/'data/quarter2.parquet';pd.read_parquet(p).to_csv(root/'data/quarter2.csv',index=False);p.unlink()
        mapping['observations'][1]['glob']='quarter2.csv'
        p=root/'data/quarter1.csv';f=pd.read_csv(p);pd.concat([f,f.iloc[:1]],ignore_index=True).to_csv(p,index=False)
        inv=data_ingest.inventory(root/'data')
        c=copy.deepcopy(DEFAULTS);c.update(output_root=str(root/'session'),data_dir=str(root/'data'),min_free_disk_gb=.001)
        Path(c['output_root']).mkdir()
        return c,mapping,inv

    def test_resume_keeps_prefix_and_matches_fresh_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c,m,inv=self.setup_case(root)
            ref=dict(c,output_root=str(root/'reference'))
            with patch.object(data_ingest,'Writer',LegacyWriter):expected=data_ingest.normalize(ref,m,inv['fingerprint'])
            db_copy=root/'staging-backup.sqlite'
            class InterruptedWriter(LegacyWriter):
                def add(self,frame):
                    super().add(frame)
                    if len(self.frames)==2:
                        self.flush();shutil.copy2(Path(c['output_root'])/'cache/staging.sqlite',db_copy)
                        raise RuntimeError('simulated stop after complete shard')
            with patch.object(data_ingest,'Writer',InterruptedWriter),self.assertRaisesRegex(RuntimeError,'simulated stop'):
                data_ingest.normalize(c,m,inv['fingerprint'])
            db=Path(c['output_root'])/'cache/staging.sqlite';shutil.copy2(db_copy,db)
            prefix=Path(c['output_root'])/'cache/rows/part-00000.parquet';prefix_hash=sha(prefix)
            cp=root/'checkpoint.json';prepare_checkpoint(c,m,inv,cp)
            actual=finish_checkpoint(c,m,inv,cp)
            self.assertEqual(prefix_hash,sha(prefix))
            for field in expected:
                if field!='files':self.assertEqual(expected[field],actual[field],field)
            old=pd.concat([pd.read_parquet(p) for p in sorted((root/'reference/cache/rows').glob('*.parquet'))],ignore_index=True)
            new=pd.concat([pd.read_parquet(p) for p in sorted((root/'session/cache/rows').glob('*.parquet'))],ignore_index=True)
            pd.testing.assert_frame_equal(old,new)
            pd.testing.assert_frame_equal(pd.read_parquet(root/'reference/cache/device_metadata.parquet'),
                                          pd.read_parquet(root/'session/cache/device_metadata.parquet'))
            self.assertEqual(actual['duplicate_rows_collapsed'],1)
            self.assertEqual(finish_checkpoint(c,m,inv,cp),actual)
            original_checkpoint=cp.read_bytes();cp.write_bytes(original_checkpoint+b' ')
            with self.assertRaisesRegex(ValueError,'Checkpoint changed'):finish_checkpoint(c,m,inv,cp)
            cp.write_bytes(original_checkpoint)
            # Restarts cannot accept edited metadata or corrupted retained output.
            (root/'session/cache/manifest.json').unlink()
            f=pd.read_parquet(prefix);f.loc[0,'history_rows']=9;f.to_parquet(prefix,index=False)
            with self.assertRaisesRegex(ValueError,'Partial device history'):finish_checkpoint(c,m,inv,cp)

    def test_unsupported_mapping_fails_before_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,m,inv=self.setup_case(Path(tmp));m['followup']['mode']='complete_coverage'
            with self.assertRaisesRegex(ValueError,'indicator'):prepare_checkpoint(c,m,inv,Path(tmp)/'checkpoint.json')


if __name__=='__main__':unittest.main()
