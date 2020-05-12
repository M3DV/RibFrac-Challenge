import argparse
import os
import shutil
from pathlib import Path
from time import perf_counter

from decompression import decompress_data


def initialize(args):
    """
    Initialize the data directory.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments received from command line.
    """
    # save the data directory in ENVIRON
    data_dir = args.data_dir
    with open("ENVIRON", "w") as f:
        f.write(args.data_dir)

    # create data_dir if it doesn't exist
    if not os.path.exists(data_dir):
        print("Data directory {} doesn't exist and is automatically \
            created.".format(data_dir))
        os.mkdir(data_dir)


def main(args):
    """
    Initialize the data directory and decompress all data.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments received from command line.
    """
    total_beg = perf_counter()
    print("Setting up directory...")
    initialize(args)

    print("Decompressing data...")
    raw_data_path = os.path.join(Path(__file__).parent.parent, "data")
    decompress_data(raw_data_path, args.data_dir)
    # copy the metadata information .csv
    shutil.copyfile(os.path.join(raw_data_path, "info.csv"),
        os.path.join(args.data_dir, "info.csv"))

    total_time = perf_counter() - total_beg
    print(f"Setup finished. Total time spent: {total_time // 60} mins.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="The directory where \
        the decompressed data will be saved.", required=True)
    args = parser.parse_args()
    main(args)
