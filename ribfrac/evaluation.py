import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from sklearn.metrics import auc
from tqdm import tqdm

from .nii_dataset import NiiDataset

__all__ = ["froc", "plot_froc", "evaluate"]


clf_conf_mat_cols = ["Buckle", "Displaced", "Nondisplaced", "Segmental",
    "FP", "Ignore"]
clf_conf_mat_rows = ["Buckle", "Displaced", "Nondisplaced", "Segmental", "FN"]


def _get_clf_result(x):
    if not pd.isna(x["gt_class"]):
        return x["gt_class"]
    elif x["hit_label"] != 0:
        return "Ignore"
    else:
        return "FP"


def _get_clf_confusion_matrix(gt_info, pred_metrics):
    conf_mat = pd.DataFrame(np.zeros((len(clf_conf_mat_rows),
        len(clf_conf_mat_cols))), index=clf_conf_mat_rows,
        columns=clf_conf_mat_cols)
    for i in range(len(pred_metrics)):
        gt_class = pred_metrics.gt_class[i]
        pred_class = pred_metrics.pred_class[i]
        conf_mat.loc[pred_class, gt_class] += 1
    
    for i in range(len(gt_info)):
        if gt_info.label_index[i] not in pred_metrics.hit_label.tolist():
            conf_mat.loc["FN", gt_info["class"][i]] += 1
    
    return conf_mat


def _compile_pred_metrics(iou_matrix, inter_matrix, gt_info, pred_info):
    """
    Compile prediction metrics into a Pandas DataFrame

    Parameters
    ----------
    iou_matrix : numpy.ndarray
        IoU array with shape of (n_pred, n_gt).
    inter_matrix : numpy.ndarray
        Intersection array with shape of (n_pred, n_gt).
    gt_info : pandas.DataFrame
        DataFrame containing GT information.
    pred_info : pandas.DataFrame
        DataFrame containing prediction information.

    Returns
    -------
    pred_metrics : pandas.DataFrame
        A dataframe of prediction metrics.
    """
    # meanings of each column:
    # pred_label --  The index of prediction
    # max_iou -- The highest IoU this prediction has with any certain GT
    # hit_label -- The GT label with which this prediction has the highest IoU
    # prob -- The confidence prediction of this prediction
    # pred_class -- The classification prediction
    # gt_class -- The ground-truth prediction
    # num_gt -- Total number of GT in this volume
    pred_metrics = pd.DataFrame(np.zeros((iou_matrix.shape[0], 4)),
        columns=["pred_label", "max_iou", "hit_label", "prob"])
    pred_metrics["pred_label"] = np.arange(1, iou_matrix.shape[0] + 1)
    pred_metrics["max_iou"] = iou_matrix.max(axis=1)
    pred_metrics["hit_label"] = iou_matrix.argmax(axis=1) + 1
    # if max_iou == 0, this prediction doesn't hit any GT
    pred_metrics["hit_label"] = pred_metrics.apply(lambda x: x["hit_label"]
        if x["max_iou"] > 0 else 0, axis=1)
    pred_metrics["prob"] = pred_info.sort_values("label_index")\
        ["probs"].values
    pred_metrics["pred_class"] = pred_info.sort_values("label_index")\
        ["class"].values
    pred_metrics = pred_metrics.merge(gt_info[["label_index", "class"]],
        how="left", left_on="hit_label", right_on="label_index")
    pred_metrics.rename({"class": "gt_class"}, axis=1, inplace=True)
    pred_metrics.drop("label_index", axis=1, inplace=True)
    pred_metrics["gt_class"] = pred_metrics.apply(_get_clf_result, axis=1)

    pred_metrics["num_gt"] = iou_matrix.shape[1]

    clf_conf_mat = _get_clf_confusion_matrix(gt_info, pred_metrics)

    return pred_metrics, clf_conf_mat


def evaluate_single_prediction(gt_label, pred_label, gt_info, pred_info):
    """
    Evaluate a single prediction.

    Parameters
    ----------
    gt_label : numpy.ndarray
        The numpy array of ground-truth labelled from 1 to n.
    pred_label : numpy.ndarray
        The numpy array of prediction labelled from 1 to n.
    gt_info : pandas.DataFrame
        DataFrame containing GT information.
    pred_info : pandas.DataFrame
        DataFrame containing prediction information.

    Returns
    -------
    pred_metrics : pandas.DataFrame
        A dataframe of prediction metrics.
    """
    gt_label = gt_label.astype(np.uint8)
    pred_label = pred_label.astype(np.uint8)
    assert gt_label.shape == pred_label.shape,\
        "The prediction and ground-truth have different shapes. gt:"\
        f" {gt_label.shape} and pred: {pred_label.shape}."

    gt_bin = (gt_label > 0).astype(np.uint8)
    pred_bin = (pred_label > 0).astype(np.uint8)

    num_gt = int(gt_label.max())
    num_pred = int(pred_label.max())
    iou_matrix = np.zeros((num_gt, num_pred))
    inter_matrix = np.zeros_like(iou_matrix)

    intersection = np.logical_and(gt_bin, pred_bin)
    union = label(np.logical_or(gt_bin, pred_bin))
    for region in regionprops(label(intersection)):
        centroid = tuple([int(round(x)) for x in region.centroid])
        gt_idx = gt_label[centroid[0], centroid[1], centroid[2]]
        pred_idx = pred_label[centroid[0], centroid[1], centroid[2]]
        union_idx = union[centroid[0], centroid[1], centroid[2]]
        if gt_idx == 0 or pred_idx == 0 or union_idx == 0:
            continue
        inter_area = region.area
        union_area = (union == union_idx).sum()
        iou = inter_area / (union_area + 1e-8)
        iou_matrix[gt_idx - 1, pred_idx - 1] = iou
        inter_matrix[gt_idx - 1, pred_idx - 1] = inter_area

    iou_matrix = iou_matrix.T
    inter_matrix = inter_matrix.T
    pred_metrics, clf_conf_mat = _compile_pred_metrics(iou_matrix,
        inter_matrix, gt_info, pred_info)

    return pred_metrics, clf_conf_mat


def _froc_single_thresh(df_list, p_thresh, iou_thresh):
    """
    Calculate the FROC for a single confidence threshold.

    Parameters
    ----------
    df_list : list of pandas.DataFrame
        List of Pandas DataFrame of prediction metrics.
    p_thresh : float
        The probability threshold of positive predictions.
    iou_thresh : float
        The IoU threshold of predictions being considered as "hit".

    Returns
    -------
    fpr : float
        False positive rate for this threshold.
    recall : float
        Recall rate for this threshold.
    """
    EPS = 1e-8

    df_list = 
    total_gt = sum([df["num_gt"][0] for df in df_list])
    df_pos_pred = [df.loc[df["prob"] >= p_thresh] for df in df_list]
    total_tp = sum([len(df.loc[df["max_iou"] > iou_thresh,
        "hit_label"].unique()) for df in df_pos_pred])
    total_fp = sum([len(df) - len(df.loc[df["max_iou"] > iou_thresh])
        for df in df_pos_pred])

    fpr = (total_fp + EPS) / (len(df_list) + EPS)
    recall = (total_tp + EPS) / (total_gt + EPS)

    return fpr, recall


def _interpolate_recall_at_fp(fpr_recall, key_fp):
    fpr_recall_less_fp = fpr_recall.loc[fpr_recall.fpr <= key_fp]
    fpr_recall_more_fp = fpr_recall.loc[fpr_recall.fpr >= key_fp]

    if len(fpr_recall_less_fp) == 0:
        return 0
    
    if len(fpr_recall_more_fp) == 0:
        return fpr_recall.recall.max()

    fpr_0 = fpr_recall_less_fp["fpr"].values[-1]
    fpr_1 = fpr_recall_more_fp["fpr"].values[0]
    recall_0 = fpr_recall_less_fp["recall"].values[-1]
    recall_1 = fpr_recall_more_fp["recall"].values[0]
    recall_at_fp = recall_0 + (recall_1 - recall_0)\
        * ((key_fp - fpr_0) / (fpr_1 - fpr_0))
    
    return recall_at_fp


def _get_key_recall(fpr, recall, key_fp):
    fpr_recall = pd.DataFrame({"fpr": fpr, "recall": recall})
    key_recall = [_interpolate_recall_at_fp(fpr_recall, fp) for fp in key_fp]

    return key_recall


def froc(df_list, iou_thresh=0.2, key_fp=(0.5, 1, 2, 4, 8)):
    """
    Calculate the FROC curve.

    Parameters
    df_list : list of pandas.DataFrame
        List of prediction metrics.
    iou_thresh : float
        The IoU threshold of predictions being considered as "hit".
    key_fp : tuple of float
        The key false positive per scan used in evaluating the sensitivity
        of the model.

    Returns
    -------
    fpr : list of float
        List of false positive rate for a range of threshold from 0 to 1.
    recall : float
        List of recall rate for a range of threshold from 0 to 1.
    """
    fpr_recall = [_froc_single_thresh(df_list, p_thresh, iou_thresh)
        for p_thresh in np.arange(0, 1, 0.01)]
    fpr = [x[0] for x in fpr_recall]
    recall = [x[1] for x in fpr_recall]
    key_recall = _get_key_recall(fpr, recall, key_fp)

    return fpr, recall, key_fp, key_recall


def plot_froc(fpr, recall):
    """
    Plot the FROC curve.

    Parameters
    ----------
    fpr : list of float
        List of false positive rates at different confidence thresholds.
    recall : list of float
        List of recall at different confidence thresholds.
    """
    _, ax = plt.subplots()
    ax.plot(fpr, recall)
    ax.set_title("FROC")
    plt.show()


def evaluate(gt_dir, pred_dir, gt_csv_path, pred_csv_path):
    """
    Evaluate predictions against the ground-truth.

    Parameters
    ----------
    gt_dir : str
        The ground-truth nii directory.
    pred_dir : str
        The prediction nii directory.
    gt_csv_path : str
        The ground-truth information csv path. The csv should contain 3
        columns: pid (patient id), label_index (GT index in the volume), and
        class (classification prediction).
    pred_csv_path : str
        The prediction information csv path. The csv should contain 3 columns:
        pid (patient id), label_index (prediction index in the volume)
        probs (prediction confidence), and class (classification prediction).

    Returns
    -------
    eval_results : list of pandas.DataFrame
        Evaluation results for each prediction.
    fpr : list of float
        List of false positive rate for a range of threshold from 0 to 1.
    recall : float
        List of recall rate for a range of threshold from 0 to 1.
    aufroc : float
        Area under FROC curve. Range from 0 to 1.
    """
    gt_iter = NiiDataset(gt_dir)
    pred_iter = NiiDataset(pred_dir)
    gt_info = pd.read_csv(gt_csv_path)
    pred_info = pd.read_csv(pred_csv_path)
    pred_info["pid"] = pred_info.pid.map(lambda x: x.strip())

    assert len(pred_iter) == len(gt_iter),\
        "Unequal number of predictions and ground-truths."
    assert [os.path.basename(x) for x in gt_iter.file_list]\
        == [os.path.basename(x) for x in pred_iter.file_list],\
        "Unmatched file names in ground-truth and prediction directory."

    eval_results = []
    for i in tqdm(range(len(gt_iter))):
        gt_arr, pid = gt_iter[i]
        cur_gt_info = gt_info.loc[gt_info.pid == pid].reset_index(drop=True)
        pred_arr, _ = pred_iter[i]
        cur_pred_info = pred_info.loc[pred_info.pid == pid, :]\
            .reset_index(drop=True)
        eval_results.append(evaluate_single_prediction(
            gt_arr, pred_arr, cur_gt_info, cur_pred_info))

    det_results = [x[0] for x in eval_results]
    clf_results = pd.DataFrame(np.sum([x[1].values for x in eval_results],
        axis=0), index=clf_conf_mat_rows, columns=clf_conf_mat_cols)
    fpr, recall, key_fp, key_recall = froc(det_results)
    plot_froc(fpr, recall)

    return fpr, recall, key_fp, key_recall, clf_results


if __name__ == "__main__":
    gt_dir = "/mnt/sdb/data/rib_frac/ribfrac_challenge/labels/val"
    pred_dir = "/mnt/sdb/data/rib_frac/labelled_model_predictions/val"
    gt_csv_path = "/mnt/sdb/data/rib_frac/val_sample_gt_info.csv"
    pred_csv_path = "/mnt/sdb/data/rib_frac/val_sample_pred_info.csv"
    fpr, recall, key_fp, key_recall, clf_results = evaluate(gt_dir, pred_dir,
        gt_csv_path, pred_csv_path)
    print(key_recall)
    print(clf_results)

