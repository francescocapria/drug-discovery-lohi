"""

Fingerprint computation

All functions return NumPy arrays of shape (n_molecules, n_features).

Computed fingerprints can be optionally cached on disk in the features/ folder 

Supported fingerprint types:
- ECFP4 (Morgan fingerprints, radius=2)
- MACCS keys (167-bit structural keys)
- RDKit topological fingerprints (path-based)
- RDKit 2D descriptors

"""

import os
import logging
from typing import List, Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

logger = logging.getLogger(__name__)


def smiles_to_mols(smiles_list: List[str]):
    mols = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning(f"Skipping invalid SMILES at index {i}: {smi[:50]}")
            mol = Chem.MolFromSmiles("C")  
        mols.append(mol)
    return mols


def compute_ecfp4(smiles_list: List[str], n_bits: int = 1024) -> np.ndarray:
    """Compute ECFP4 (Morgan radius=2) fingerprints."""
    mols = smiles_to_mols(smiles_list)
    X = np.zeros((len(mols), n_bits), dtype=np.uint8)
    gen = GetMorganGenerator(radius=2, fpSize=n_bits)

    for i, mol in enumerate(mols):
        fp = gen.GetFingerprintAsNumPy(mol)
        X[i] = fp

    return X


def compute_maccs(smiles_list: List[str]) -> np.ndarray:
    """Compute MACCS keys fingerprints."""
    mols = smiles_to_mols(smiles_list)
    X = np.zeros((len(mols), 167), dtype=np.uint8)

    for i, mol in enumerate(mols):
        fp = MACCSkeys.GenMACCSKeys(mol)
        DataStructs.ConvertToNumpyArray(fp, X[i])

    return X


def compute_rdkit_descriptors(smiles_list: List[str]) -> np.ndarray:
    mols = smiles_to_mols(smiles_list)
    X = np.array([list(Descriptors.CalcMolDescriptors(mol).values()) for mol in mols], dtype=np.float64)

    for j in range(X.shape[1]):
        col = X[:, j]
        mask = ~np.isfinite(col)
        if mask.any():
            median = np.nanmedian(col)
            col[mask] = median if not np.isnan(median) else 0.0

    X = np.clip(X, -1e15, 1e15)

    return X


def compute_rdkit_topo(
    smiles_list: List[str],
    min_path: int = 1,
    max_path: int = 7,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute RDKit topological (path-based) fingerprints."""
    mols = smiles_to_mols(smiles_list)
    X = np.zeros((len(mols), n_bits), dtype=np.uint8)

    for i, mol in enumerate(mols):
        fp = Chem.RDKFingerprint(
            mol,
            minPath=min_path,
            maxPath=max_path,
            fpSize=n_bits,
        )
        DataStructs.ConvertToNumpyArray(fp, X[i])

    return X


def compute_fingerprints(
    smiles_list: List[str],
    fp_type: str,
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute molecular fingerprints with optional caching.
    
    """
    if cache_path is not None and os.path.exists(cache_path):
        logger.info(f"Loading fingerprints from cache: {cache_path}")
        data = np.load(cache_path)
        return data["X"]

    if fp_type == "ecfp4":
        X = compute_ecfp4(smiles_list)
    elif fp_type == "maccs":
        X = compute_maccs(smiles_list)
    elif fp_type == "rdkit_topo":
        X = compute_rdkit_topo(smiles_list)
    elif fp_type == "rdkit_desc":
        X = compute_rdkit_descriptors(smiles_list)
    else:
        raise ValueError(
            "fp_type must be one of: 'ecfp4', 'maccs', 'rdkit_topo', 'rdkit_desc'"
        )

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, X=X)
        logger.info(f"Saved fingerprint cache to: {cache_path}")

    return X