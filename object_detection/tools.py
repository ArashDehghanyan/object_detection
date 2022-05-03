import os
import xml.etree.ElementTree as Et
import copy
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import Sequence
from object_detection.constants import *


def parse_annotation(ann_dir, img_dir, labels=[]):
    """Extract object position and labels exist in the image."""
    all_img = []
    seen_labels = {}

    for ann in sorted(os.listdir(ann_dir)):
        if "xml" not in ann:
            continue
        img = {'object': []}
        tree = Et.parse(ann_dir + ann)
        for elm in tree.iter():
            if 'filename' in elm.tag:
                path_to_img = img_dir + elm.text
                img['filename'] = path_to_img
                # make sure that file exists
                if not os.path.exists(path_to_img):
                    assert False, "File does not exist:\n {}".format(path_to_img)
            if 'width' in elm.tag:
                img['width'] = int(elm.text)
            if 'height' in elm.tag:
                img['height'] = int(elm.text)
            if 'object' in elm.tag or 'part' in elm.tag:
                obj = {}
                for attr in list(elm):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text
                        if len(labels) > 0 and obj['name'] not in labels:
                            continue
                        else:
                            img['object'] += [obj]
                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))
        if len(img['object']) > 0:
            all_img += [img]

    return all_img, seen_labels


class ImageReader(object):

    def __init__(self, image_h, image_w, norm=None):
        self.IMAGE_H = image_h
        self.IMAGE_W = image_w
        self.norm = norm

    def encode_core(self, image, reorder_rgb=True):
        image = cv2.resize(image, (self.IMAGE_H, self.IMAGE_W))

        if reorder_rgb:
            image = image[:, :, ::-1]

        if self.norm:
            image = self.norm(image)

        return image

    def fit(self, train_instance):

        if not isinstance(train_instance, dict):
            train_instance = {'filename': train_instance}
        img_name = train_instance['filename']
        img = cv2.imread(img_name)
        # Check if image is not Null
        if img is None:
            print("{} not found!".format(img_name))
        h, w, c = img.shape
        img = self.encode_core(img, reorder_rgb=True)

        if "object" in train_instance.keys():
            all_objs = copy.deepcopy(train_instance['object'])
            for obj in all_objs:
                for attr in ['xmin', 'xmax']:
                    obj[attr] = int(obj[attr] * float(self.IMAGE_W) / w)
                    obj[attr] = max(min(obj[attr], self.IMAGE_W), 0)

                for attr in ['ymin', 'ymax']:
                    obj[attr] = int(obj[attr] * float(self.IMAGE_H) / h)
                    obj[attr] = max(min(obj[attr], self.IMAGE_H), 0)
        else:
            return img
        return img, all_objs


class BoundBox:

    def __init__(self, xmin, ymin, xmax, ymax, confidence=None, classes=None):
        self.xmin, self.ymin = xmin, ymin
        self.xmax, self.ymax = xmax, ymax
        self.confidence = confidence
        self.set_classes(classes)

    def set_classes(self, classes):
        self.classes = classes
        self.label = np.argmax(classes)

    def get_label(self):
        return self.label

    def get_score(self):
        return self.classes[self.label]


class BestAnchorBoxFinder(object):

    def __init__(self, ANCHORS):
        self.anchors = [BoundBox(0, 0, ANCHORS[2*i], ANCHORS[2*i+1])
                        for i in range(int(len(ANCHORS) // 2))]

    def _interval_overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        return max(min(x2, x4) - max(x1, x3), 0)

    def intersect_over_union(self, box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
        intersect_area = intersect_w * intersect_h

        w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
        w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
        union_area = w1 * h1 + w2 * h2 - intersect_area

        return float(intersect_area) / (union_area + 1e-6)

    def find(self, center_w, center_h):
        """Find best anchor that fits this box"""
        best_anchor = -1
        max_iou = -1

        shifted_box = BoundBox(0, 0, center_w, center_h)
        for i in range(len(self.anchors)):
            anchor = self.anchors[i]
            iou = self.intersect_over_union(shifted_box, anchor)
            if max_iou < iou:
                best_anchor = i
                max_iou = iou

        return best_anchor, max_iou


def rescale_center_xy(obj, config):
    """
    Converts a pixel original value to range between [0, GRID_H] and [0, GRID_W]
    :param obj: a dict containing xmin, ymin, xmax, ymax
    :param config: a dict containing IMAGE_W, GRID_W, IMAGE_H, GRID_H
    :return: center_x and center_y
    """
    center_x = 0.5 * (obj['xmin'] + obj['xmax'])
    center_x = center_x / (float(config['IMAGE_W']) / config['GRID_W'])
    center_y = 0.5 * (obj['ymin'] + obj['ymax'])
    center_y = center_y / (float(config['IMAGE_H']) / config['GRID_H'])
    return center_x, center_y


def rescale_center_wh(obj, config):
    """
    Convert image width and height values to range [0, GRID_W/GRID_H]
    :param obj: a dict containing xmin, ymin, xmax, ymax
    :param config: a dict containing IMAGE_H, GRID_H, IMAGE_W, GRID_W
    :return: center_w, center_h
    """
    center_w = (obj['xmax'] - obj['xmin']) / (float(config['IMAGE_W']) / config['GRID_W'])
    center_h = (obj['ymax'] - obj['ymin']) / (float(config['IMAGE_H']) / config['GRID_H'])

    return center_w, center_h


class SimpleBatchGenerator(Sequence):
    """Define custom batch generator."""
    def __init__(self, images, config, norm=None, shuffle=True):
        self.config = config
        self.config['BOX'] = int(len(self.config['ANCHORS'])/2)
        self.config['CLASS'] = len(self.config['LABELS'])
        self.images = images
        self.best_anchorbox_finder = BestAnchorBoxFinder(self.config['ANCHORS'])
        self.image_reader = ImageReader(self.config['IMAGE_H'],
                                        self.config['IMAGE_W'],
                                        norm=norm)
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images)) / self.config['BATCH_SIZE']))

    def __getitem__(self, idx):
        """
        Get a batch of images and their labels
        :param idx: non-negative integet value (batch index)
        :return: x_batch: a numpy array of shape (batch_size, image_h, image_w, nb channels)
                 y_batch: a numpy array of shape (batch_size, grid_h, grid_w, box, 4+1+nb classes)
                 box: the number of anchor boxes
                 b_batch: a numpy array of shape (batch_size, 1, 1, 1, TRUE_BOX_BUFFER, 4)
        """
        # left bound
        l_bound = idx * self.config['BATCH_SIZE']
        # right bound
        r_bound = (idx + 1) * self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
        # instantiate values
        instance_count = 0
        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],
                            self.config['GRID_W'], self.config['BOX'], 4+1+self.config['CLASS']))
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.config['TRUE_BOX_BUFFER'], 4))

        for train_instance in self.images[l_bound:r_bound]:
            img, all_objs = self.image_reader.fit(train_instance)

            true_box_index = 0
            for obj in all_objs:
                if obj['xmin'] < obj['xmax'] and obj['ymin'] < obj['ymax'] and obj['name'] in self.config['LABELS']:
                    center_x, center_y = rescale_center_xy(obj, self.config)

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                        object_index = self.config['LABELS'].index(obj['name'])
                        center_w, center_h = rescale_center_wh(obj, self.config)
                        box = [center_x, center_y, center_w, center_h]
                        best_anchor, max_iou = self.best_anchorbox_finder.find(center_w, center_h)

                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4] = 1.    # ground truth confidence
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5+object_index] = 1    # class probability of the object

                        b_batch[instance_count, 0, 0, 0, true_box_index] = box
                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

            x_batch[instance_count] = img
            instance_count += 1
        return x_batch, [y_batch, b_batch]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images)


def check_object_in_grid_anchor_pair(irow, y_batch):
    for igrid_h in range(generator_config['GRID_H']):
        for igrid_w in range(generator_config['GRID_W']):
            for ianchor in range(generator_config['BOX']):
                vec = y_batch[irow, igrid_h, igrid_w, ianchor, :]
                C = vec[4]  # Ground truth confidence
                if C == 1:
                    class_nm = np.array(LABELS)[np.where(vec[5:])]
                    assert len(class_nm) == 1
                    print("igrid_h={:2.0f}, igrid_w={:2.0f}, iAnchor={:2.0f}, {}".format(
                        igrid_h, igrid_w, ianchor, class_nm[0]))


def plot_image_with_grid_cell_partition(irow, x_batch):
    img = x_batch[irow]
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    # plot grid on image
    for wh in ['W', 'H']:
        grid_ = generator_config['GRID_' + wh]  # 13
        image_ = generator_config['IMAGE_' + wh]    # 416
        if wh == "W":
            pltax = plt.axvline     # plot vertical lines
            plttick = plt.xticks
        else:
            pltax = plt.axhline     # plot horizontal lines
            plttick = plt.yticks

        for i in range(grid_):
            pltax(image_*i/grid_, color='yellow', alpha=0.3)
        plttick([image_*(j+0.5)/grid_ for j in range(grid_)],
                ["iGrid_{}={}".format(wh, j) for j in range(grid_)])


def plot_grid(irow, y_batch):
    import seaborn as sns
    color_pallete = list(sns.xkcd_rgb.values())
    iobj = 0
    for igrid_h in range(generator_config['GRID_H']):
        for igrid_w in range(generator_config['GRID_W']):
            for ianchor in range(generator_config['BOX']):
                vec = y_batch[irow, igrid_h, igrid_w, ianchor, :]
                confidence = vec[4]
                if confidence == 1:
                    class_nm = np.array(LABELS)[np.where(vec[5:])]
                    x, y, w, h = vec[:4]
                    multx = generator_config['IMAGE_W'] / generator_config['GRID_W']
                    multy = generator_config['IMAGE_H'] / generator_config['GRID_H']
                    xmin = x - 0.5 * w
                    xmax = x + 0.5 * w
                    ymin = y - 0.5 * h
                    ymax = y + 0.5 * h
                    c = color_pallete[iobj]
                    iobj += 1
                    # plot center X
                    plt.text(multx * x, multy * y, 'X', color=c, fontsize=23)
                    # plot rectangle
                    plt.plot(
                        np.array([xmin, xmin])*multx, np.array([ymin, ymax])*multy, color=c, linewidth=10
                    )
                    plt.plot(
                        np.array([xmin, xmax])*multx, np.array([ymin, ymin])*multy, color=c, linewidth=10
                    )
                    plt.plot(
                        np.array([xmax, xmax])*multx, np.array([ymin, ymax])*multy, color=c, linewidth=10
                    )
                    plt.plot(
                        np.array([xmin, xmax])*multx, np.array([ymax, ymax])*multy, color=c, linewidth=10
                    )


class WeightReader:
    """Read weights from a weight file"""
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def reset(self):
        self.offset = 4


class OutputRescaler(object):

    def __init__(self, anchors):
        self.ANCHORS = anchors

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def _softmax(self, x, axis=-1, t=-100):
        x = x - np.max(x)

        if np.min(x) < t:
            x = x / np.min(x) * t
        e_x = np.exp(x)
        return e_x / e_x.sum(axis, keepdims=True)

    def get_shifting_matrix(self, netout):

        grid_h, grid_w, box = netout.shape[:3]
        no = netout[..., 0]

        anchor_width = self.ANCHORS[::2]
        anchor_length = self.ANCHORS[1::2]

        mat_grid_w = np.zeros_like(no)
        for i_grid_w in range(grid_w):
            mat_grid_w[:, i_grid_w, :] = i_grid_w

        mat_grid_h = np.zeros_like(no)
        for i_grid_h in range(grid_h):
            mat_grid_h[i_grid_h, :, :] = i_grid_h

        mat_anchor_w = np.zeros_like(no)
        for ianchor in range(box):
            mat_anchor_w[:, :, ianchor] = anchor_width[ianchor]

        mat_anchor_h = np.zeros_like(no)
        for ianchor in range(box):
            mat_anchor_h[:, :, ianchor] = anchor_length[ianchor]

        return mat_grid_w, mat_grid_h, mat_anchor_w, mat_anchor_h

    def fit(self, netout):
        '''
        netout  : np.array of shape (N grid h, N grid w, N anchor, 4 + 1 + N class)

        a single image output of model.predict()
        '''
        grid_h, grid_w, box = netout.shape[:3]
        (mat_grid_w, mat_grid_h, mat_anchor_w, mat_anchor_h) = self.get_shifting_matrix(netout)

        # Bounding box parameters
        netout[..., 0] = (self._sigmoid(netout[..., 0]) + mat_grid_w) / grid_w  # x
        netout[..., 1] = (self._sigmoid(netout[..., 1]) + mat_grid_h) / grid_w  # y
        netout[..., 2] = (np.exp(netout[..., 2]) * mat_anchor_w) / grid_w     # width
        netout[..., 3] = (np.exp(netout[..., 3]) * mat_anchor_h) / grid_h     # height
        # Rescale the confidence to range between 0 and 1
        netout[..., 4] = self._sigmoid(netout[..., 4])
        expanded_conf = tf.expand_dims(netout[..., 4], axis=-1)
        netout[..., 5:] = expanded_conf * self._softmax(netout[..., 5:])
        return netout


def find_high_class_probability_bbox(netout_scale, obj_threshold):

    grid_h, grid_w, box = netout_scale.shape[:3]
    boxes = []

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(box):
                classes = netout_scale[row, col, b, 5:]

                if np.sum(classes) > 0:
                    x, y, w, h = netout_scale[row, col, b, :4]
                    confidence = netout_scale[row, col, b, 4]
                    _box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
                    if _box.get_score() > obj_threshold:
                        boxes.append(_box)
    return boxes


def draw_box(image, boxes, labels, baseline=0.05, verbose=False):
    """Draws a rectangle around the detected object and writes object class with its probability."""
    def adjust_minmax(c, _max):
        return min(max(c, 0), _max)

    image = copy.deepcopy(image)
    image_h, image_w, _ = image.shape
    score_rescaled = np.array([box.get_score() for box in boxes])
    score_rescaled /= baseline

    colors = sns.color_palette("husl", 8)
    for sr, color, box in zip(score_rescaled, colors, boxes):
        xmin = adjust_minmax(int(box.xmin * image_w), image_w)
        ymin = adjust_minmax(int(box.ymin * image_h), image_h)
        xmax = adjust_minmax(int(box.xmax * image_w), image_w)
        ymax = adjust_minmax(int(box.ymax * image_h), image_h)

        text = "{:18} {:04.0f}".format(labels[box.label], box.get_score())
        if verbose:
            print("{} XMIN={:4.0f}, YMIN={:4.0f}, XMAX={:4.0f}, YMAX={:4.0f}".format(
                text, xmin, ymin, xmax, ymax
            ))

        image = cv2.rectangle(
            image,
            pt1=(xmin, ymin),
            pt2=(xmax, ymax),
            color=color,
            thickness=int(sr)
        )

        image = cv2.putText(
            image,
            text=text,
            org=(xmin+13, ymin+13),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1e-3 * image_h,
            color=(1, 0, 1),
            thickness=1
        )

    return image


def nonmax_suppresion(boxes, iou_threshold, obj_threshold):
    """Reduce the box number."""
    best_anchor_box_finder = BestAnchorBoxFinder([])
    CLASS = len(boxes[0].classes)
    index_boxes = []

    for c in range(CLASS):
        class_probability_from_bbox = [box.classes[c] for box in boxes]

        sorted_indices = list(reversed(np.argsort(class_probability_from_bbox)))
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue
            else:
                index_boxes.append(index_i)
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    bbox_iou = best_anchor_box_finder.intersect_over_union(boxes[index_i], boxes[index_j])
                    if bbox_iou >= iou_threshold:
                        classes = boxes[index_j].classes
                        classes[c] = 0
                        boxes[index_j].set_classes(classes)

    new_boxes = [boxes[i] for i in index_boxes if boxes[i].get_score() > obj_threshold]
    return new_boxes
