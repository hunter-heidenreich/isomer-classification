from dataclasses import dataclass

import h5py
import omegaconf
import torch
from torch.utils.data import Dataset


@dataclass
class IsomerDataConfig:
    """Data configuration for the Isomer dataset."""

    # Path to the dataset
    pth: str

    # Load the dataset in memory
    in_memory: bool = True


class IsomerDataset(Dataset):
    """Isomer dataset class for PyTorch DataLoader."""

    def __init__(self, cfg: omegaconf.DictConfig):
        self.cfg = IsomerDataConfig(**cfg.dataset)
        self.data = self._load_data()

    def _load_data(self):
        data = []
        if self.cfg.in_memory:
            with h5py.File(self.cfg.pth, "r") as file:
                for _, chem_formula_group in file.items():
                    for isomer_smiles, isomer_group in chem_formula_group.items():
                        symbols = [
                            symbol.decode("utf-8")
                            for symbol in isomer_group["symbols"][:]
                        ]
                        conformers_positions = isomer_group["conformers"][:]
                        iupac_name = isomer_group.attrs["IUPACName"]

                        conformers_positions_tensor = torch.tensor(
                            conformers_positions, dtype=torch.float32
                        )

                        isomer_data = {
                            "smiles": isomer_smiles,
                            "iupac_name": iupac_name,
                            "symbols": symbols,
                            "conformers_positions": conformers_positions_tensor,
                        }
                        data.append(isomer_data)
        else:
            raise NotImplementedError("Loading data from disk is not implemented yet.")

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    test_cfg = {
        "dataset": {"pth": "data/isomer_datasets/C4H10-10000.h5", "in_memory": True}
    }
    dataset = IsomerDataset(omegaconf.OmegaConf.create(test_cfg))
    print(dataset.data[0])
