from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Union

import h5py
import numpy as np
import omegaconf
import torch
from ase import Atoms
from dscribe.descriptors import CoulombMatrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from tqdm import tqdm


class FeatureTransform(Enum):
    """Feature transformation for the Isomer dataset."""
    NULL = 0  # No transformation
    COULOMB_MATRIX_EIGENVALUES = 1  # Eigenvalues of the Coulomb matrix
    COULOMB_MATRIX_L2_SORTED = 2  # L2 sorted Coulomb matrix
    COULOMB_MATRIX_RANDOM = 3  # Random permutation of the Coulomb matrix
    

def to_ase(atoms: np.ndarray, symbols: Sequence[str]) -> Atoms:
    """Convert the input data to ASE Atoms object.
    
    Args:
        atoms: np.ndarray of shape (n_atoms, 3) with the atomic positions
        symbols: Sequence of atomic symbols
        
    Returns:
        ASE Atoms object
    """
    return Atoms(
        symbols=symbols,
        positions=atoms,
    )
    

def feature_transform(
    x: np.ndarray,
    symbols: Sequence[str],
    transform: FeatureTransform,
    sigma: Optional[float] = None,
) -> torch.Tensor:
    """Apply feature transformation to the input data.
    
    Args:
        x: np.ndarray of shape (n_atoms, 3) with the atomic positions
        symbols: Sequence of atomic symbols
        transform: FeatureTransform enum
        
    Returns:
        torch.Tensor with the transformed features
    """
    if transform == FeatureTransform.NULL:
        x = torch.tensor(x, dtype=torch.float32)
        return x.flatten()
    
    n_atoms = len(symbols)
    atoms = to_ase(x, symbols)
    if transform == FeatureTransform.COULOMB_MATRIX_EIGENVALUES:
        cm = CoulombMatrix(
            n_atoms_max=n_atoms,
            permutation="eigenspectrum",
        )
        cm_eigenvalues = cm.create(atoms)
        return torch.tensor(cm_eigenvalues, dtype=torch.float32)
    elif transform == FeatureTransform.COULOMB_MATRIX_L2_SORTED:
        cm = CoulombMatrix(
            n_atoms_max=n_atoms,
            permutation="sorted_l2",
        )
        cm_l2_sorted = cm.create(atoms)
        cm_l2_sorted = torch.tensor(cm_l2_sorted, dtype=torch.float32)
        cm_l2_sorted = cm_l2_sorted.reshape(n_atoms, n_atoms)
        
        cm_l2_unique = torch.empty(n_atoms * (n_atoms + 1) // 2, dtype=torch.float32)
        k = 0
        for i in range(n_atoms):
            for j in range(i, n_atoms):
                cm_l2_unique[k] = cm_l2_sorted[i, j]
                k += 1
        return cm_l2_sorted
    elif transform == FeatureTransform.COULOMB_MATRIX_RANDOM:
        cm = CoulombMatrix(
            n_atoms_max=n_atoms,
            permutation="random",
            sigma=sigma,
        )
        cm_random = cm.create(atoms)
        cm_random = torch.tensor(cm_random, dtype=torch.float32)
        cm_random = cm_random.reshape(n_atoms, n_atoms)
        
        cm_random_unique = torch.empty(n_atoms * (n_atoms + 1) // 2, dtype=torch.float32)
        k = 0
        for i in range(n_atoms):
            for j in range(i, n_atoms):
                cm_random_unique[k] = cm_random[i, j]
                k += 1
        return cm_random
    else:
        raise ValueError(f"Unknown feature transformation: {transform}")


@dataclass
class IsomerDataConfig:
    """Data configuration for the Isomer dataset."""

    # Path to the dataset
    pth: str

    # Load the dataset in memory
    in_memory: bool = True
    
    # Feature transformation
    feature_transform: FeatureTransform = FeatureTransform.NULL
    
    # Random permutation of the Coulomb matrix
    sigma: Optional[float] = None


class IsomerDataset(Dataset):
    """Isomer dataset class for PyTorch DataLoader."""

    def __init__(
        self,
        cfg: Union[IsomerDataConfig, dict],
    ):
        if isinstance(cfg, dict):
            cfg = IsomerDataConfig(**cfg)
        self.cfg = cfg

        # atom type symbols
        self.symbols = None

        # conversion between names and smiles
        self.name2smiles = {}
        self.smiles2name = {}

        # raw positions
        self.x = []

        # conformer classification labels
        self.y = []

        # Load the data
        self._load_data()

        self.le = LabelEncoder()
        self.le.fit(self.y)
        self.data = list(zip(self.x, self.le.transform(self.y)))

    def _load_data(self):
        if self.cfg.in_memory:
            with h5py.File(self.cfg.pth, "r") as file:
                for _, chem_formula_group in file.items():
                    for isomer_smiles, isomer_group in tqdm(chem_formula_group.items()):
                        # Load and validate atomic symbols
                        symbols = [
                            symbol.decode("utf-8")
                            for symbol in isomer_group["symbols"][:]
                        ]
                        if self.symbols is None:
                            self.symbols = symbols
                        else:
                            assert (
                                symbols == self.symbols
                            ), "All isomers should have the same atom types"

                        # Log the conversion between names and smiles
                        iupac_name = isomer_group.attrs["IUPACName"]
                        self.name2smiles[iupac_name] = isomer_smiles
                        self.smiles2name[isomer_smiles] = iupac_name

                        # Ingest the conformers positions
                        conformers_positions = isomer_group["conformers"][:]
                        for conformer in conformers_positions:
                            self.x.append(conformer)
                            self.y.append(iupac_name)
        else:
            raise NotImplementedError("Loading data from disk is not implemented yet.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        
        x = feature_transform(
            x,
            self.symbols,
            self.cfg.feature_transform,
            sigma=self.cfg.sigma,
        )
        
        return x, y


if __name__ == "__main__":
    test_cfg = {
        "pth": "data/isomer_datasets/C4H10-10000.h5", 
        "in_memory": True,
        "feature_transform": FeatureTransform.COULOMB_MATRIX_RANDOM,
        "sigma": 0.1,
    }
    dataset = IsomerDataset(omegaconf.OmegaConf.create(test_cfg))
    print(len(dataset))
    print(dataset[0])
    print(dataset[-1])
