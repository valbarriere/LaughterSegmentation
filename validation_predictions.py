"""
Only works for ru/en because of the normalize_name...
But this can be adapted quickly
"""
import glob
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath("/home/vbarrier/standup_comedyclub/src"))
from normalize_name import normalize_filename

def calculate_iou(interval1, interval2):
    """
    Calculate Intersection over Union (IoU) for two intervals.
    Each interval is a tuple (start, end).
    """
    start1, end1 = interval1
    start2, end2 = interval2

    # Calculate intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)

    # Calculate union
    area1 = end1 - start1
    area2 = end2 - start2
    union = area1 + area2 - intersection

    # Avoid division by zero
    if union == 0:
        return 0.0

    return intersection / union

def evaluate_predictions_with_iou(predictions, labels, iou_threshold=0.5):
    """
    Evaluate predictions against labels using IoU.
    Returns counts of True Positives (TP), False Positives (FP), and False Negatives (FN).
    """
    TP = 0
    FP = 0
    FN = 0

    # Track which labels have been matched
    matched_labels = set()
    not_matched_labels = set()

    # Check each prediction
    for pred in predictions:
        best_iou = 0.0
        best_label_idx = -1

        for i, label in enumerate(labels):
            iou = calculate_iou(pred, label)
            if iou > best_iou:
                best_iou = iou
                best_label_idx = i

        if best_iou >= iou_threshold:
            TP += 1
            matched_labels.add(best_label_idx)  # Mark this label as matched
        else:
            FP += 1
            # not_matched_labels.add(label)

    # Check for FN
    for i, label in enumerate(labels):
        for pred in predictions:
            iou = calculate_iou(pred, label)
            if iou > best_iou:
                best_iou = iou
                best_label_idx = i

        if best_iou < iou_threshold:
            not_matched_labels.add(label)

    # Count False Negatives (labels not matched by any prediction)
    # FN = len(labels) - len(matched_labels)
    FN = len(not_matched_labels)

    return TP, FP, FN, not_matched_labels

def _test():
    # Example usage
    predictions = [(1, 3), (5, 7), (9, 11)]  # List of prediction intervals
    labels = [(2, 4), (6, 8), (10, 12)]       # List of label intervals
    iou_threshold = 0.5  # Set the IoU threshold

    TP, FP, FN = evaluate_predictions_with_iou(predictions, labels, iou_threshold)
    print(f"True Positives: {TP}")
    print(f"False Positives: {FP}")
    print(f"False Negatives: {FN}")

def calculate_metrics(TP, FP, FN):
    """
    Calculate precision, recall, and accuracy given TP, FP, and FN.
    """
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    f1 = 2*TP/(2*TP+FP+FN)

    return precision, recall, accuracy, f1

def read_K_annot(tsv_file):

    df = pd.read_csv(tsv_file, sep="\t", header=None)

    # Filter rows where the first column is "laughter"
    laughter_df = df[df[0] == "laughter"]

    # Extract start and end times
    start_times = laughter_df[2].tolist()
    end_times = laughter_df[3].tolist()

    # Create a new DataFrame in the desired format
    return pd.DataFrame([start_times, end_times], index=["start_sec", "end_sec"])

def df2list(df):
    return list(zip(df.T['start_sec'], df.T['end_sec']))

if __name__ == "__main__":

    list_gt = glob.glob("data/standup/annotations_K/en/*")
    iou_threshold = 0.3
    
    for name in list_gt:
        basename = os.path.basename(name).split('.')[0]
        # print(name, normalize_filename(basename))
        # exit
        json_name = '/home/vbarrier/LaughterSegmentation/output/' + normalize_filename(basename)+'_Instruments'*False+'.json'
        # print(json_name)
        dfpred = pd.read_json(json_name)
        dfgt = read_K_annot(name)

        TP, FP, FN, not_matched_labels = evaluate_predictions_with_iou(df2list(dfpred), df2list(dfgt), iou_threshold=iou_threshold)
        precision, recall, accuracy, f1= calculate_metrics(TP, FP, FN)
        print('****'*5, json_name, '****'*5)
        print( basename, TP, FP, FN, not_matched_labels)
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"F1: {f1:.2f}")
    