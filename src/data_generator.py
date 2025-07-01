from os import path
from typing import Optional, Union
from constants import LabelIdx
import numpy as np


def get_dataset(
        is_classification: bool = False,
        set_name: str = "",
        is_sim: bool = False,
        target_label: Optional[LabelIdx] = None,
        subdir: Optional[str] = None
) -> tuple[Union[np.ndarray, list], Union[np.ndarray, list], Union[np.ndarray, list], Union[np.ndarray, list]]:
    """
    Get a dataset for training or testing, classification or regression
    :param subdir: Subdirectory which contains the data. Assumed to be under root/compiled_datasets/{type}/.
    :param target_label: Target label type (e.g. node traversal vs. route iter time)
    :param is_classification: Is this classification? (else regression)
    :param set_name: Dataset name (prefix)
    :param is_sim: Is this for routing simulation?
    :return:
    """
    assert set_name != "", f"Blank set name"
    assert not is_sim, f"UNSUPPORTED!"
    # Get path to data directory
    base_path = path.dirname(__file__)
    if not is_sim:
        if is_classification:
            target_dir = "classification"
        else:
            target_dir = "regression"
        if subdir is None:
            data_path = path.abspath(
                path.join(
                    base_path, "..", "compiled_datasets", target_dir
                )
            )
        else:
            data_path = path.abspath(
                path.join(
                    base_path, "..", "compiled_datasets", target_dir, subdir
                )
            )
    else:
        data_path = path.abspath(
            path.join(
                base_path, "..", "routing_sim_data"
            )
        )

    feature_data_path = path.abspath(
        path.join(
            data_path, f"{set_name}_X.csv"
        )
    )
    label_data_path = path.abspath(
        path.join(
            data_path, f"{set_name}_y.csv"
        )
    )
    if is_sim:
        time_data_path = path.abspath(
            path.join(
                data_path, f"{set_name}_time.csv"
            )
        )
        time_data = np.genfromtxt(time_data_path, delimiter=',', dtype=float, skip_header=1)
    else:
        time_data = None
    X = np.genfromtxt(feature_data_path, delimiter=',', dtype=float, skip_header=1)
    if target_label == LabelIdx.TOT_PURE_ROUTE_TIME:
        reg_dtype = float
    else:
        reg_dtype = int
    y = np.genfromtxt(label_data_path, delimiter=',', dtype=reg_dtype, skip_header=1)
    with open(feature_data_path, 'rt') as infile:
        header = infile.readline()
        feature_names = header.strip().split(',')
    if len(X) == 0 or len(y) == 0:
        return [], [], [], []
    if is_sim:
        if len(time_data) == 0:
            return [], [], [], []
    assert len(X) == len(y), f"{len(X)} == {len(y)}"
    if target_label is None:
        return X, y, feature_names, time_data
    else:
        return X, y[:, target_label], feature_names, time_data
