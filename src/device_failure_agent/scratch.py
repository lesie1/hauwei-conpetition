"""Delete only the disposable matrix of a finished worker, never data or models."""
from pathlib import Path
import stat


def cleanup_matrix(folder):
    root=Path(folder).resolve()
    target=root/'train_matrix.npy'
    if not target.exists() and not target.is_symlink():return 0
    if target.is_symlink() or not stat.S_ISREG(target.stat().st_mode):
        raise ValueError('Refusing non-regular scratch file: '+str(target))
    size=target.stat().st_size
    target.unlink()
    return size
