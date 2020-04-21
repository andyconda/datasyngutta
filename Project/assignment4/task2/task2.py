import numpy as np
import matplotlib.pyplot as plt
import json
import copy
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE

    # Are the boxes overlapping?
    # If no:
    if (gt_box[0] > prediction_box[2] or gt_box[2] < prediction_box[0]
        or gt_box[1] > prediction_box[3] or gt_box[3] < prediction_box[1]):
        iou = 0

    #If yes:
    else:
        # Compute intersection
        xmin = max((prediction_box[0], gt_box[0]))
        xmax = min((prediction_box[2], gt_box[2]))
        ymin = max((prediction_box[1], gt_box[1]))
        ymax = min((prediction_box[3], gt_box[3]))

        width = xmax - xmin
        height = ymax - ymin

        intersection = width * height

        # Compute union
        area_prediction = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])
        area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

        union = area_gt + area_prediction - intersection

        iou = intersection / union
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if (num_tp + num_fp) == 0:
        return 1

    precision = num_tp / (num_tp + num_fp)
    return precision


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if (num_tp + num_fn) == 0:
        return 0

    recall = num_tp / (num_tp + num_fn)
    return recall


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    ious = np.ndarray((gt_boxes.shape[0], prediction_boxes.shape[0]))
    for i in range (0, gt_boxes.shape[0]):
        for j in range (0, len(prediction_boxes)):
            ious[i, j] = calculate_iou(prediction_boxes[j], gt_boxes[i])

    # Empty list to fill
    paired_predictions = []
    paired_gt = []

    if ious.shape[1] > 0:
        # Finding prediction index for each maximum iou
        indices = np.argmax(ious, axis=1)
        maxes = np.amax(ious, axis=1)

        # Filling in lists
        for i in range(0, len(indices)):
            if maxes[i] >= iou_threshold:
                paired_predictions.append(prediction_boxes[indices[i]])
                paired_gt.append(gt_boxes[i])

    # List to np.array
    paired_predictions = np.array(paired_predictions)
    paired_gt = np.array(paired_gt)

    return paired_predictions, paired_gt


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    # Calculating every iou
    if len(prediction_boxes) > 0:
        ious = np.ndarray((gt_boxes.shape[0], prediction_boxes.shape[0]))
        for i in range(0, gt_boxes.shape[0]):
            for j in range(0, prediction_boxes.shape[0]):
                ious[i, j] = calculate_iou(prediction_boxes[j], gt_boxes[i])

        # Finding best matches
        indices = np.argmax(ious, axis=1)

        # Counting TP
        TP = 0
        for i in range(0, len(indices)):
            if ious[i, indices[i]] >= iou_threshold:
                TP += 1
    else:
        TP = 0

    #Calculating FP and FN
    FP = prediction_boxes.shape[0] - TP
    FN = gt_boxes.shape[0] - TP

    # Creating dictionary
    result = {"true_pos": TP, "false_pos": FP, "false_neg": FN}
    return result


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    # Calculating total TP, FP and FN
    total_TP = 0
    total_FP = 0
    total_FN = 0
    for i in range(0, len(all_gt_boxes)):
        result = calculate_individual_image_result(all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)
        total_TP = total_TP + result["true_pos"]
        total_FP = total_FP + result["false_pos"]
        total_FN = total_FN + result["false_neg"]

    # Calculating precision and recall
    if total_TP == 0:
        precision = 1
    else:
        precision = total_TP / (total_TP + total_FP)
    recall = total_TP / (total_TP + total_FN)
    return (precision, recall)


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        tuple: (precision, recall). Both float.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE
    precisions = []
    recalls = []

    for t in confidence_thresholds:
        all_pred_over_tresh = []
        for i in range(0, len(all_prediction_boxes)):
            im_pred_over_thresh_list = []
            for j in range(0, all_prediction_boxes[i].shape[0]):
                if confidence_scores[i][j] >= t:
                    im_pred_over_thresh_list.append(all_prediction_boxes[i][j])
            im_pred_over_thresh_array = np.array(im_pred_over_thresh_list)
            all_pred_over_tresh.append(im_pred_over_thresh_array)

        precision_and_recall = calculate_precision_recall_all_images(all_pred_over_tresh, all_gt_boxes, iou_threshold)
        precisions.append(precision_and_recall[0])
        recalls.append(precision_and_recall[1])

    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    eleven_precisions = []
    for level in recall_levels:
        i = 0
        copy_precisions = precisions.copy()
        copy_recalls = recalls.copy()
        while copy_recalls[copy_recalls.shape[0] - 1] < level:
            copy_recalls = np.delete(copy_recalls, copy_recalls.shape[0] - 1)
            copy_precisions = np.delete(copy_precisions, copy_precisions.shape[0] - 1)
            i += 1
            if copy_recalls.shape[0] == 0:
                break
        if copy_precisions.shape[0] == 0:
            eleven_precisions.append(0)
        else:
            eleven_precisions.append(np.amax(copy_precisions))
    print(eleven_precisions)
    average_precision = np.average(eleven_precisions)
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
