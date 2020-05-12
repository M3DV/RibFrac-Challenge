import gzip
import os
import shutil
import zipfile


def _extract_zip(src, dst):
    """
    Extract contents from a .zip file.

    Parameters
    ----------
    src : str
        The source path to extract contents from.
    dst : str
        The destination path to save the extracted files.
    """
    # check if src is a valid .zip
    assert zipfile.is_zipfile(src), "{} is not a valid .zip file.".format(src)

    zip_file = zipfile.ZipFile(src, "r")
    for file in zip_file.namelist():
        zip_file.extract(file, dst)


def _extract_gz(src, dst):
    """
    Extract contents from a .gz file.

    Parameters
    -----------
    src : str
        The source path from which to extract contents.
    dst : str
        The destination path to save the extracted files.
    """
    assert src.endswith(".gz"), "{} is not a valid .gz file.".format(src)
    assert os.path.exists(src), "{} does not exist.".format(src)

    with gzip.open(src, "rb") as f_src:
        # xxx.postfix.gz --> xxx.postfix
        file_name = os.path.basename(src)[:-3]
        with open(os.path.join(dst, file_name), "wb") as f_dst:
            shutil.copyfileobj(f_src, f_dst)


def _cat_multi_vol_zip(src, dst):
    """
    Concatenate volumes of .zip file together for further extraction.

    Parameters
    ----------
    src : str
        The source path where .zip volumes are stored.
    dst : str
        The destination path to save the concatenated .zip.
    """
    concat_cmd = "zip -s 0 {} --out {}".format(src, dst)
    os.system(concat_cmd)


def _extract_multi_vol_zip(src, dst):
    """
    Extract multiple continuous volumes of .zip file.

    Parameters
    ----------
    src : str
        The source path where .zip volumes are stored.
    dst : str
        The destination path to save the extracted content.
    """
    cat_zip_path = os.path.join(dst, os.path.basename(src))
    _cat_multi_vol_zip(src, cat_zip_path)
    _extract_zip(cat_zip_path, dst)
    os.remove(cat_zip_path)


def _extract_all_gz_in_dir(directory):
    """
    Extract all .gz files under one directory.

    Parameters
    ----------
    directory : str
        The directory where all .gz files resides
    """
    for file in os.listdir(directory):
        if file.endswith(".gz"):
            gz_fname = os.path.join(directory, file)
            _extract_gz(gz_fname, os.path.dirname(gz_fname))
            os.remove(gz_fname)


def _create_folder(path):
    """
    Create a folder.

    Parameters
    ----------
    path : str
        The folder to be created.
    """
    if not os.path.exists(path):
        os.mkdir(path)


def _create_layout(root_dir, subsets):
    """
    Create the folder layout for the competition data.

    Parameters
    ----------
    root_dir : str
        The directory to store competition data.
    subset : list of str or tuple of str
        Subsets of data in competiton.
    """
    _create_folder(os.path.join(root_dir, "images"))
    _create_folder(os.path.join(root_dir, "labels"))

    for subset in subsets:
        _create_folder(os.path.join(root_dir, "images", subset))
        _create_folder(os.path.join(root_dir, "labels", subset))


def decompress_data(src, dst):
    """
    Decompress all competition data.

    Parameters
    ----------
    data_dir : str
        The directory where the compressed data is saved.
    dst : str
        The directory where decompressed data will be saved.
    """
    assert os.path.exists(src), "{} does not exist. Please download the \
        entire repository and keep it as it originally is".format(src)

    # create folder layout at the destination folder
    subset_list = ["train", "val", "test"]
    _create_layout(dst, subset_list)

    # extract data
    for subset in subset_list:
        subset_img_src = os.path.join(src, "images", subset + ".zip")
        subset_img_dst = os.path.join(dst, "images", subset)
        _extract_multi_vol_zip(subset_img_src, subset_img_dst)
        _extract_all_gz_in_dir(subset_img_dst)

        subset_lbl_src = os.path.join(src, "labels", subset + ".zip")
        subset_lbl_dst = os.path.join(dst, "labels", subset)
        _extract_zip(subset_lbl_src, subset_lbl_dst)
        _extract_all_gz_in_dir(subset_lbl_dst)

        print("Finished decompressing {}.".format(subset))
