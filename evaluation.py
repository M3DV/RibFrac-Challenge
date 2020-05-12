import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from tqdm import tqdm


def _distance(p1, p2):
    """
    Return the Euclidean distance between two points.

    Parameters
    ----------
    p1, p2 : numpy.ndarray
        The numpy array of point coordinates.

    Returns
    -------
    dist : scalar
        The distance between p1 and p2.
    """
    dist = np.linalg.norm(p1 - p2)

    return dist


def _compile_pred_metrics(iou_matrix, inter_matrix, pred, pred_label):
    """
    Compile prediction metrics into a Pandas DataFrame

    Parameters
    ----------
    iou_matrix : numpy.ndarray
        IoU array with shape of (n_pred, n_gt).
    inter_matrix : numpy.ndarray
        Intersection array with shape of (n_pred, n_gt).
    pred : numpy.ndarray
        A numpy array of probability prediction for each voxel.
    pred_label : numpy.ndarray
        A numpy array containing n GTs labelled from 1 to n.

    Returns
    -------
    pred_metrics : pandas.DataFrame
        A dataframe of prediction metrics.
    """
    # meanings of each column:
    # pred_label --  The index of prediction
    # max_iou -- The highest IoU this prediction has with any certain GT
    # hit_label -- The GT label with which this prediction has the highest IoU
    # intersect_vox -- Number of voxels contained in the intersection between
    #     this prediction and its hit GT
    # prob -- The confidence prediction of this prediction
    # vol -- This prediction's volume in pixels
    # z, y, x -- Coordinates of this prediction's centroid
    # num_gt -- Total number of GT in this volume
    pred_metrics = pd.DataFrame(np.zeros((iou_matrix.shape[0], 9)),
        columns=["pred_label", "max_iou", "hit_label", "intersect_vox",
        "prob", "vol", "z", "y", "x"])
    pred_metrics["pred_label"] = np.arange(1, iou_matrix.shape[0] + 1)
    pred_metrics["max_iou"] = iou_matrix.max(axis=1)
    pred_metrics["hit_label"] = iou_matrix.argmax(axis=1) + 1
    pred_metrics["intersect_vox"] = np.choose(iou_matrix.argmax(axis=1),
        inter_matrix.T)
    # if max_iou == 0, this prediction doesn't hit any GT
    pred_metrics["hit_label"] = pred_metrics.apply(lambda x: x["hit_label"]
        if x["max_iou"] > 0 else 0, axis=1)
    region_props = regionprops(pred_label, pred)
    pred_metrics["prob"] = [prop.mean_intensity for prop in region_props]
    pred_metrics["vol"] = [prop.area for prop in region_props]
    pred_metrics[["z", "y", "x"]] = np.array([prop.centroid
        for prop in region_props])
    pred_metrics["num_gt"] = iou_matrix.shape[1]

    return pred_metrics


def evaluate_single_prediction(pred, gt_label, threshold):
    """
    Evaluate a single prediction.

    Parameters
    ----------
    pred : numpy.ndarray
        The numpy array of prediction.
    gt_label : numpy.ndarray
        The numpy array of ground-truth.
    threshold : float
        If a voxel's prediction probability is higher than threshold,
        it is treated as a positive prediction

    Returns
    -------
    pred_metrics : pandas.DataFrame
        A dataframe of prediction metrics.
    """
    gt_label = gt_label.astype(int)
    gt_bin = gt_label > 0

    # Label connected regions
    pred_bin = pred > threshold
    pred_label = label(pred_bin)

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
    pred_metrics = _compile_pred_metrics(iou_matrix, inter_matrix, pred,
        pred_label)

    return pred_metrics


def evaluate(pred_iter, gt_iter, threshold=0.1):
    """
    Evaluate predictions against the ground-truth.

    Parameters
    ----------
    pred_iter : sequence or iterable of numpy.ndarray
        An iterator or sequence of numpy array containing predictions.
    gt_iter : sequence or iterable of numpy.ndarray
        An iterator or sequence of numpy array containing ground-truths.

    Returns
    -------
    eval_results : list of pandas.DataFrame
        Evaluation results for each prediction.
    """
    assert len(pred_iter) == len(gt_iter),\
        "Unequal number of predictions and ground-truths."

    eval_results = []
    for i in tqdm(range(len(gt_iter))):
        eval_results.append(evaluate_single_prediction(pred_iter[i],
            gt_iter[i], threshold))

    return eval_results


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

    total_gt = total_tp = total_fp = 0
    for df in df_list:
        if len(df) != 0:
            gt_cnt = df["num_gt"][0]
            df = df[df["prob"] >= p_thresh]
            tp_cnt = len(df.loc[df["max_iou"] > iou_thresh,
                "hit_label"].unique())
            fp_cnt = len(df) - len(df.loc[df["max_iou"] > iou_thresh])

            total_gt += gt_cnt
            total_tp += tp_cnt
            total_fp += fp_cnt

    fpr = total_fp / (len(df_list) + EPS)
    recall = total_tp / (total_gt + EPS)

    return fpr, recall


def froc(df_list, iou_thresh=0):
    """
    Calculate the FROC curve.

    Parameters
    df_list : list of pandas.DataFrame
        List of prediction metrics.
    iou_thresh : float
        The IoU threshold of predictions being considered as "hit".

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

    return fpr, recall
