import numpy as np
import tensorflow as tf
from object_detection.constants import *


def get_cell_grid(grid_w, grid_h, batch_size, box):
    """Helper function to assure that the bounding box x and y are in the grid cell scale"""
    # cell_x shape = (1, 13, 13, 1, 1)
    cell_x = tf.cast(tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1, 1)), tf.float32)
    # cell_y shape = (1, 13, 13, 1, 1)
    cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
    # cell_grid shape = (batch_size, 13, 13, box, 2)
    cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [batch_size, 1, 1, box, 1])
    return cell_grid


def adjust_scale_predictions(y_pred, cell_grid, anchors):
    """Adjust prediction."""
    box = int(len(anchors) / 2)
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid   # bx, by
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(anchors, [1, 1, 1, box, 2]) # bw, bh
    pred_box_conf = tf.sigmoid(y_pred[..., 4])  # box confidence
    pred_box_class = y_pred[..., 5:]    # adjust class probabilities

    return pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class


def extract_ground_truth(y_true):
    true_box_xy = y_true[..., :2]
    true_box_wh = y_true[..., 2:4]
    true_box_conf = y_true[..., 4]
    true_box_class = tf.argmax(y_true[..., 5:], -1)
    return true_box_xy, true_box_wh, true_box_conf, true_box_class


def calc_loss_xywh(true_box_conf, coord_scale, true_box_xy, pred_box_xy, true_box_wh, pred_box_wh):
    """Calculate coordination loss."""
    coord_mask = tf.expand_dims(true_box_conf, axis=-1) * coord_scale
    coord_box_nb = tf.reduce_sum(tf.cast(coord_mask > 0.0, tf.float32))
    loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (coord_box_nb + 1e-6) / 2.
    loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (coord_box_nb + 1e-6) / 2.
    return loss_xy + loss_wh, coord_mask


def calc_loss_class(true_box_conf, class_scale, true_box_class, pred_box_class):
    """Calculate class loss."""
    class_mask = true_box_conf * class_scale
    class_box_nb = tf.reduce_sum(tf.cast(class_mask > 0.0, tf.float32))

    class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=true_box_class, logits=pred_box_class
    )
    class_loss = tf.reduce_sum(class_loss * class_mask) / (class_box_nb + 1e-6)

    return class_loss


def get_iou(true_xy, true_wh, pred_xy, pred_wh):
    """Calculate intersect area."""
    true_wh_half = 0.5 * true_wh
    true_xy_min = true_xy - true_wh_half
    true_xy_max = true_xy + true_wh_half

    pred_wh_half = 0.5 * pred_wh
    pred_xy_min = pred_xy - pred_wh_half
    pred_xy_max = pred_xy + pred_wh_half

    intersect_min = tf.maximum(true_xy_min, pred_xy_min)
    intersect_max = tf.minimum(true_xy_max, pred_xy_max)
    intersect_wh = tf.maximum(intersect_max - intersect_min, 0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_area = true_wh[..., 0] * true_wh[..., 1]
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    union_area = true_area + pred_area - intersect_area + 1e-6
    iou_score = tf.truediv(intersect_area, union_area)
    return iou_score


def calc_iou_pred_true_assigned(true_box_conf, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
    iou_scores = get_iou(true_box_xy, true_box_wh, pred_box_xy, pred_box_wh)
    true_box_conf_iou = iou_scores * true_box_conf
    return true_box_conf_iou


def calc_iou_pred_true_best(pred_box_xy, pred_box_wh, true_box_xy, true_box_wh):
    """Finds the IOU of the objects that most likely included (best fitted)."""
    true_xy = tf.expand_dims(true_box_xy, 4)    # (N BATCH, GRID_H, GRID_W, N ANCHOR, 1, 2)
    true_wh = tf.expand_dims(true_box_wh, 4)

    pred_xy = tf.expand_dims(pred_box_xy, 4)    # (N BATCH, GRID_H, GRID_W, N ANCHOR, 1, 2)
    pred_wh = tf.expand_dims(pred_box_wh, 4)
    # (N BATCH, GRID_H, GRID_W, N ANCHOR, 1)
    iou_scores = get_iou(true_xy, true_wh, pred_xy, pred_wh)
    best_iou = tf.reduce_max(iou_scores, axis=4)    # (N BATCH, GRID_H, GRID_W, N ANCHOR)
    return best_iou


def get_conf_mask(best_ious, true_box_conf, true_box_conf_iou, no_object_scale, object_scale):
    """Get confidence mask."""
    conf_mask = tf.cast(best_ious < 0.6, tf.float32) * (1 - true_box_conf) * no_object_scale
    conf_mask += true_box_conf_iou * object_scale
    return conf_mask


def calc_conf_loss(conf_mask, true_box_conf_iou, pred_box_conf):
    """Calculate confidence loss."""
    conf_box_nb = tf.reduce_sum(tf.cast(conf_mask > 0.0, tf.float32))
    conf_loss = tf.reduce_sum(tf.square(true_box_conf_iou - pred_box_conf) * conf_mask) / (conf_box_nb + 1e-6) / 2.
    return conf_loss


def custom_yolo_loss(y_true, y_pred):
    """Custom yolo v2 loss function."""
    # total_recall = tf.Variable(0.)

    # Step 1: adjust prediction output
    cell_grid = get_cell_grid(GRID_W, GRID_H, len(y_true), BOX)
    pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class = adjust_scale_predictions(y_pred, cell_grid, ANCHORS)

    # Step 2: extract ground truth output
    true_box_xy, true_box_wh, true_box_conf, true_box_class = extract_ground_truth(y_true)

    # Step 3: Calculate loss for bounding box parameters
    loss_xywh, coord_mask = calc_loss_xywh(true_box_conf, LAMBDA_COORD, true_box_xy, pred_box_xy, true_box_wh, pred_box_wh)

    # Step 4: Calculate loss for class probabilities
    loss_class = calc_loss_class(true_box_conf, LAMBDA_CLASS, true_box_class, pred_box_class)
    # Step 5: For each (grid cell, anchor) pair, calculate the IoU between predicted and ground truth bounding box
    true_box_conf_iou = calc_iou_pred_true_assigned(true_box_conf, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh)
    # Step 6: For each predicted bounded box from (grid cell, anchor box),
    # calculate the best IOU, regardless of the ground truth anchor box
    # that each object gets assigned.
    best_ious = calc_iou_pred_true_best(pred_box_xy, pred_box_wh, true_box_xy, true_box_wh)
    # Step 7: For each grid cell, calculate the L_{i,j}^{noobj}
    conf_mask = get_conf_mask(best_ious, true_box_conf, true_box_conf_iou, LAMBDA_NO_OBJECT, LAMBDA_OBJECT)
    # Step 8: Calculate loss for the confidence
    loss_conf = calc_conf_loss(conf_mask, true_box_conf_iou, pred_box_conf)

    loss = loss_xywh + loss_class + loss_conf
    return loss
