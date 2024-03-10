from argparse import ArgumentParser

from pysomer.data.gen import get_isomer_dataset


def main():
    parser = ArgumentParser(
        description="Create a dataset of conformers for a given molecule"
    )
    parser.add_argument(
        "chem_formula",
        type=str,
        help="Chemical formula of the molecule (e.g. C1H4 (methane))",
    )
    parser.add_argument(
        "--num_confs",
        type=int,
        default=10,
        help="Number of conformers desired for each isomer",
    )
    args = parser.parse_args()

    get_isomer_dataset(args.chem_formula, args.num_confs)


if __name__ == "__main__":
    main()
