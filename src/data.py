from pathlib import Path 
import h5py
import numpy as np 
import pandas as pd 
import logging

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw"

def inspect_h5(path) -> None:
    "Schema discovery for h5 file"
    with h5py.File(path, "r") as f:
        logger.info("File: %s", path)
        logger.info("Top-level keys: %s", list(f.keys()))
        logger.info("All datasets:")

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                logger.info("  %-20s shape=%-20s dtype=%s", name, str(obj.shape), obj.dtype)

        f.visititems(visit)

def _decode_var_names(arr) -> list[str]:
    """Var name decoder, bytes->str"""
    return [v.decode() if isinstance(v, bytes) else str(v) for v in np.array(arr).ravel()]

def load_split(path, split: str = "dev", include_virtual: bool = False, include_health: bool = False) -> pd.DataFrame:
    """
    Load one split of N-CMAPSS DS02 into a single DataFrame
    Args:
        path: path to H5 file
        split: {"dev", "test"}, dev = 6 training engines, "test" 3 held-out engines,
        include_virtual: virtual sensors, util only for analysis
        include_health; health param, same as aforementioned 
    Returns:
        DataFrame: tabular representation of a singular split 
    """

    if split not in ("dev", "test"):
        raise ValueError(f"Split must be 'dev' or 'test', got: {split}")
    
    logger.info("Loading split=%s from %s", split, Path(path).name)
    logger.debug("include_virtual=%s, include_health=%s",include_virtual, include_health)

    with h5py.File(path, "r") as f:
        W_var  = _decode_var_names(f["W_var"][:])
        Xs_var = _decode_var_names(f["X_s_var"][:])
        Xv_var = _decode_var_names(f["X_v_var"][:])
        T_var  = _decode_var_names(f["T_var"][:])
        A_var  = _decode_var_names(f["A_var"][:])

        W  = np.array(f[f"W_{split}"],   dtype=np.float32)
        Xs = np.array(f[f"X_s_{split}"], dtype=np.float32)
        A  = np.array(f[f"A_{split}"],   dtype=np.float32)
        Y  = np.array(f[f"Y_{split}"],   dtype=np.float32).reshape(-1, 1)

        blocks = [W, Xs]
        cols = W_var + Xs_var

        if include_virtual:
            blocks.append(np.array(f[f"X_v_{split}"], dtype=np.float32))
            cols += Xv_var
        if include_health:
            blocks.append(np.array(f[f"T_{split}"], dtype=np.float32))
            cols += T_var

        blocks += [A, Y]
        cols += A_var + ["RUL"]
    
    df = pd.DataFrame(np.hstack(blocks), columns=cols)
    for c in ("unit", "cycle"):
        if c in df.columns:
            df[c] = df[c].astype(np.int32)
    
    mem_gb = df.memory_usage(deep=True).sum() / 1e9
    logger.info("Loaded split=%s: shape=%s, memory=%.2f GB", split, df.shape, mem_gb)

    return df 

def print_var_names(path) -> dict:
    """Read all *_var arrays and return as a dict, also logging each one"""
    var_keys = ["W_var", "X_s_var", "X_v_var", "T_var", "A_var"]
    with h5py.File(path, "r") as f:
        names = {k: _decode_var_names(f[k][:]) for k in var_keys}
    for k, v in names.items():
        logger.info("%-10s (%2d): %s", k, len(v), v)
    return names