import numpy as np

ANCHORS = np.array(
    [
        1.07709888,  1.78171903,  # anchor box 1, width , height
        2.71054693,  5.12469308,  # anchor box 2, width,  height
        10.47181473, 10.09646365,  # anchor box 3, width,  height
        5.48531347,  8.11011331    # anchor box 4, width, height
    ]
)

LABELS = [
    'aeroplane',  'bicycle', 'bird',  'boat',      'bottle',
    'bus',        'car',      'cat',  'chair',     'cow',
    'diningtable','dog',    'horse',  'motorbike', 'person',
    'pottedplant','sheep',  'sofa',   'train',   'tvmonitor'
]

GRID_H, GRID_W = 13, 13
IMAGE_H, IMAGE_W = 416, 416
BATCH_SIZE = 4
TRUE_BOX_BUFFER = 50
BOX = int(len(ANCHORS) / 2)
CLASS = len(LABELS)

generator_config = {
    'IMAGE_H'         : IMAGE_H,
    'IMAGE_W'         : IMAGE_W,
    'GRID_H'          : GRID_H,
    'GRID_W'          : GRID_W,
    'BOX'             : BOX,
    'LABELS'          : LABELS,
    'ANCHORS'         : ANCHORS,
    'BATCH_SIZE'      : BATCH_SIZE,
    'TRUE_BOX_BUFFER' : TRUE_BOX_BUFFER,
}

LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (0, 0.01),
    (75, 0.001),
    (105, 0.0001),
]

LAMBDA_NO_OBJECT = 1.0
LAMBDA_OBJECT = 5.0
LAMBDA_COORD = 1.0
LAMBDA_CLASS = 1.0

OBJ_THRESHOLD = 0.5
IOU_THRESHOLD = 0.01
