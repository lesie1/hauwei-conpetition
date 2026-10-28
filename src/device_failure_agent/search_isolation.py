"""Process-wide raw/TEST read blocks and read-only protection of prior experiments."""
import json
import os
from pathlib import Path
import sys


def install():
    path = os.environ.get('TYPEA_SEARCH_ISOLATION')
    if not path:
        return
    spec = json.loads(Path(path).read_text())
    forbidden = [Path(x).resolve() for x in spec['forbidden_reads']]
    protected = [Path(x).resolve() for x in spec['readonly_roots']]
    def check(event,args):
        if event != 'open' or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        p=Path(os.fsdecode(args[0])).resolve()
        if 'cache_test' in p.parts or any(p==x or p.is_relative_to(x) for x in forbidden):
            raise PermissionError('Raw/TEST access forbidden for exploratory search')
        mode,flags=args[1] or '',args[2]
        if any(p==x or p.is_relative_to(x) for x in protected):
            if any(x in mode for x in 'wax+') or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):
                raise PermissionError('Prior experiment is read-only')
    sys.addaudithook(check)
