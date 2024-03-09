import json
import logging
import os
from typing import List

import h5py
import numpy as np
import requests
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


MAYGEN_PATH = os.environ.get(
    "MAYGEN_PATH",
    os.path.join(os.path.dirname(__file__), "../../plugin/MAYGEN-1.8.jar"),
)
DATA_DIR = os.environ.get(
    "DATA_DIR", os.path.join(os.path.dirname(__file__), "../../data")
)

logger = logging.getLogger(__name__)


def gen_isomer_smiles(
    chem_formula: str,
) -> List[str]:
    """Generate isomer SMILES for a given chemical formula.

    Calls the MAYGEN-1.8.jar plugin to generate isomer SMILES for a given chemical formula.

    Args:
        chem_formula (str): Chemical formula of the molecule

    Returns:
        List[str]: List of isomer SMILES
    """
    # Ensure the output directory exists
    out_dir = os.path.join(DATA_DIR, "isomer_smiles")
    os.makedirs(out_dir, exist_ok=True)

    # Check if the isomer SMILES are already available
    out_pth = os.path.join(out_dir, f"{chem_formula}.smi")
    try:
        with open(out_pth, "r") as f:
            out = f.readlines()
    except FileNotFoundError:
        cmd_str = f"java -jar {MAYGEN_PATH} -v -m -f {chem_formula} -smi -o {out_dir}"
        logger.info(f"Running command: {cmd_str}")
        os.system(cmd_str)

        with open(out_pth, "r") as f:
            out = f.readlines()

    out = [smi.strip() for smi in out if smi.strip()]
    logger.info(f"Read {len(out):,} isomer SMILES for {chem_formula} from file")

    return out


def get_iupac_name(smiles: str) -> str:
    """Get IUPAC name for a given SMILES string.

    Args:
        smiles (str): SMILES string of the molecule

    Returns:
        str: IUPAC name of the molecule

    Raises:
        Exception: If the request fails, raises an exception with the status code
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/IUPACName/JSON"
    logger.info(f"Requesting IUPAC name for {smiles} from {url}")
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data["PropertyTable"]["Properties"][0]["IUPACName"]
    else:
        raise Exception(f"Error: {response.status_code}")


def get_iupac_names(
    chem_formula: str,
    smiles: List[str],
) -> List[str]:
    """Get IUPAC names for a list of SMILES strings.

    Implements basic caching to avoid redundant requests.

    Args:
        chem_formula (str): Chemical formula of the molecule
        smiles (List[str]): List of SMILES strings

    Returns:
        tuple[List[str], List[str]]: List of SMILES and corresponding IUPAC names,
            sorted by SMILES

    Raises:
        Exception: If the request fails, raises an exception with the status code
    """
    # Ensure the output directory exists
    out_dir = os.path.join(DATA_DIR, "iupac_names")
    os.makedirs(out_dir, exist_ok=True)

    # Load existing IUPAC names
    out_pth = os.path.join(out_dir, f"{chem_formula}.json")
    try:
        with open(out_pth, "r") as f:
            iupac_names = json.load(f)
    except FileNotFoundError:
        iupac_names = {}

    if iupac_names:
        logger.info(f"Loaded IUPAC names for {len(iupac_names)} SMILES from file")

    # Filter out SMILES for which IUPAC names are already available
    new_smiles = [smi for smi in smiles if smi not in iupac_names]

    # Get IUPAC names for the remaining SMILES
    new_iupac_names = {}
    for smi in new_smiles:
        try:
            new_iupac_names[smi] = get_iupac_name(smi)
        except Exception as e:
            logger.error(f"Failed to get IUPAC name for {smi}: {e}")
    logger.info(f"Got IUPAC names for {len(new_iupac_names)} new SMILES")

    # Update the IUPAC names and save to file
    iupac_names.update(new_iupac_names)
    if new_iupac_names:
        with open(out_pth, "w") as f:
            json.dump(iupac_names, f)
        logger.info(f"Saved IUPAC names for {len(iupac_names)} SMILES to file")

    # Sort the SMILES and IUPAC names by SMILESå
    _smis = list(iupac_names.keys())
    _iupacs = [iupac_names[smi] for smi in _smis]
    _joint = list(zip(_smis, _iupacs))
    _joint.sort(key=lambda x: x[0])
    _smis, _iupacs = zip(*_joint)

    return _smis, _iupacs


def sample_conformers(
    smiles_str: str,
    num_confs: int = 10,
    optimize_3d: bool = False,
    random_seed: int = -1,
):
    """Takes a SMILES string and generates 3D conformers for the molecule.
    Returns ase.Atoms object with the conformers.

    Args:
    - smiles_str (str): A SMILES string representing a molecule.
    - num_confs (int): Number of conformers to generate. Default is 10.
    - optimize_3d (bool): Whether to optimize the 3D coordinates of the molecule.
        Default is False.
    - random_seed (int): Random seed for 3D coordinate generation. Default is -1 (random).

    Returns:
    - mol (rdkit.Chem.Mol): An RDKit mol object representing the molecule.
    """
    # Convert SMILES string to RDKit mol object
    mol = Chem.MolFromSmiles(smiles_str)

    # Add hydrogens to the molecule
    mol = Chem.AddHs(mol)

    # Generate 3D conformers for the molecule
    if num_confs == 1:
        AllChem.EmbedMolecule(mol, randomSeed=random_seed)
    else:
        AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=random_seed)

    # Optimize 3D coordinates of the molecule (if required)
    if optimize_3d:
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)

    # Convert RDKit mol object to ase.Atoms object
    conformers = []
    for conf_id in range(mol.GetNumConformers()):
        conf = mol.GetConformer(conf_id)
        coords = conf.GetPositions()
        conformer = Atoms(
            symbols=[atom.GetSymbol() for atom in mol.GetAtoms()], positions=coords
        )
        conformers.append(conformer)

    logger.info(f"Generated {len(conformers)} conformers for {smiles_str}")

    return conformers


def get_isomer_dataset(
    chem_formula: str,
    num_confs: int = 10,
) -> str:
    """Generate isomer dataset for a given chemical formula.

    Dataset stored as an HDF5 file with the following structure:
    - /isomers/
        - /<smiles_0>/  # SMILES string of the first isomer
            - /symbols  # Atomic symbols
            - /conformer_0  # 3D coordinates of the first conformer
            - /conformer_1  # 3D coordinates of the second conformer
            ...
        - /<smiles_1>/  # SMILES string of the second isomer
            - /symbols  # Atomic symbols
            - /conformer_0  # 3D coordinates of the first conformer
            - /conformer_1  # 3D coordinates of the second conformer
            ...
        ...

    Args:
        chem_formula (str): Chemical formula of the molecule
        num_confs (int): Number of conformers desired for each isomer. Default is 10.

    Returns:
        str: Path to the dataset file
    """
    # Generate isomer SMILES
    isomer_smiles = gen_isomer_smiles(chem_formula)

    # Get IUPAC names
    isomer_smiles, iupac_names = get_iupac_names(chem_formula, isomer_smiles)

    # Ensure the output directory exists
    out_dir = os.path.join(DATA_DIR, "isomer_datasets")
    os.makedirs(out_dir, exist_ok=True)
    out_pth = os.path.join(out_dir, f"{chem_formula}-{num_confs}.h5")

    # Sample conformers for each isomer
    with h5py.File(out_pth, "w") as h5f:
        chem_formula_grp = h5f.create_group(chem_formula)
        for i, smi in enumerate(tqdm(isomer_smiles, desc="Sampling conformers")):
            isomer_grp = chem_formula_grp.create_group(smi)
            isomer_grp.attrs["IUPACName"] = iupac_names[i]

            # Sample conformers
            conformers = sample_conformers(smi, num_confs=num_confs)

            # Assuming the first conformer's symbols represent all others (they should be identical)
            symbols = [atom.symbol for atom in conformers[0]]
            _ = isomer_grp.create_dataset(
            "symbols", data=np.array(symbols, dtype="S")
            )  # Store symbols as fixed-length byte strings

            # Storing conformers' positions
            all_positions = np.array([conformer.positions for conformer in conformers])
            _ = isomer_grp.create_dataset(
            "conformers", data=all_positions, compression="gzip", compression_opts=9
            )
        
        h5f.flush()

    return out_pth


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    # test_mol = "C4H10"
    # test_mol = "C5H12"
    # test_mol = "C6H14"
    # test_mol = "C7H16"
    # test_mol = "C8H18"
    test_mol = "C9H20"

    # num_confs = 10
    # num_confs = 100
    # num_confs = 1_000
    num_confs = 10_000

    pth = get_isomer_dataset(test_mol, num_confs=num_confs)
    print(pth)
