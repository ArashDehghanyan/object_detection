import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import copy
import xml
import os
import tarfile
from object_detection.tools import *
from object_deteection.constants import *
from object_detection.models import yolo_v2, CustomLearningRateScheduler, lr_schedule
from object_detection.loss import custom_yolo_loss


train_image_folder = "./VOCdevkit/VOC2012/JPEGImages/"
train_annot_folder = "./VOCdevkit/VOC2012/Annotations/"
path_to_weight = "./yolov2.weights"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(tf.__version__)
    print(tf.config.list_physical_devices())
    # open dataset tar file
    if not os.path.exists("VOCdevkit"):
        file = tarfile.open("C:\\Users\\MegaSystem\\Downloads\\VOCtrainval_11-May-2012.tar")
        file.extractall(os.getcwd())
        file.close()

    train_images, seen_labels = parse_annotation(
        train_annot_folder,
        train_image_folder,
        labels=LABELS
    )
    print("train number: {}".format(len(train_images)))

    # sample = train_images[0]
    #
    # print("*"*30)
    # print("Input")
    # for k, v in sample.items():
    #     print("\t{}: {}".format(k, v))
    # print("*"*30)
    # print("Output")
    # inputEncoder = ImageReader(416, 416, norm=lambda x: x/255.)
    # image, all_objects = inputEncoder.fit(sample)
    # print("\t{}".format(all_objects))
    # plt.imshow(image)
    # plt.title("image.shape={}".format(image.shape))
    # plt.axis('off')
    # plt.show()
    # print('*'*30)
    train_batch_generator = SimpleBatchGenerator(train_images[:10], generator_config,
                                                 norm=lambda x: x/255., shuffle=True)
    # for X, y in train_batch_generator:
    #     print(X.shape)
    #     print(y[0].shape, y[1].shape)

    # print("x_batch shape = {}".format(x_batch.shape))
    # print("y_batch shape = {}".format(y_batch.shape))
    # print("b_batch shape = {}".format(b_batch.shape))

    # for iframe in range(5, 10):
    #     print('-'*40)
    #     check_object_in_grid_anchor_pair(iframe, y_batch)
    #     plot_image_with_grid_cell_partition(iframe, x_batch)
    #     plot_grid(iframe, y_batch)
    #     plt.show()

    model = yolo_v2()
    # # model.summary()
    #
    # y_pred = model(X)

    # print(model.get_layer('conv_1').get_weights()[0].shape)
    weight_reader = WeightReader(path_to_weight)
    print("All weights shape = {}".format(weight_reader.all_weights.shape))
    # Set pre-trained weights to the model
    weight_reader.reset()
    conv_number = 23
    for i in range(1, conv_number + 1):
        conv_layer = model.get_layer('conv_' + str(i))
        if i < conv_number:
            norm_layer = model.get_layer('norm_' + str(i))
            size = np.prod(norm_layer.get_weights()[0].shape)

            beta = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean = weight_reader.read_bytes(size)
            var = weight_reader.read_bytes(size)
            weights = norm_layer.set_weights([gamma, beta, mean, var])

        if len(conv_layer.get_weights()) > 1:
            bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel, bias])
        else:
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel])

    last_conv_layer = model.layers[-2]  # Last convolution layer
    weights = last_conv_layer.get_weights()
    # print(weights[0].shape)
    new_kernel = np.random.normal(size=weights[0].shape) / (GRID_H * GRID_W)
    new_bias = np.random.normal(size=weights[1].shape) / (GRID_H * GRID_W)
    # last_conv_layer.set_weights([new_kernel, new_bias])

    # print(get_cell_grid(GRID_W, GRID_H, BATCH_SIZE, BOX))
    # true_boxes = tf.Variable(np.zeros_like(b_batch), dtype='float32')


    #
    # early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
    # optimizer = Adam(learning_rate=5e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
    # model.compile(loss=custom_yolo_loss, optimizer=optimizer)
    # # tf.config.run_functions_eagerly(True)
    #
    # model.fit(
    #     train_batch_generator,
    #     steps_per_epoch=len(train_batch_generator),
    #     epochs=3,
    #     verbose=1,
    #     callbacks=[
    #         # CustomLearningRateScheduler(lr_schedule),
    #         early_stop],
    #     max_queue_size=3
    # )

    image_reader = ImageReader(IMAGE_H, IMAGE_W, norm=lambda img: img/255.)
    out = image_reader.fit(train_image_folder + "/2007_005430.jpg")
    # print(out.shape)
    X_test = tf.expand_dims(out, axis=0)
    # print(X_test.shape)
    y_pred = model.predict(X_test)
    # print(y_pred.shape)
    netout = y_pred[0]
    output_rescaler = OutputRescaler(anchors=ANCHORS)
    netout_scale = output_rescaler.fit(netout)

    boxes = find_high_class_probability_bbox(netout_scale, OBJ_THRESHOLD)

    print("obj_threshold={}".format(OBJ_THRESHOLD))
    print("In total, YOLO can produce GRID_H * GRID_W * BOX = {} bounding boxes ".format(GRID_H * GRID_W * BOX))
    print("I found {} bounding boxes with top class probability > {}".format(len(boxes), OBJ_THRESHOLD))

    figsize = (12, 12)
    X_test = np.array(X_test)
    print(X_test.shape)
    ima = draw_box(X_test[0], boxes, LABELS, verbose=True)
    plt.figure(figsize=figsize)
    plt.imshow(ima)
    plt.title("Plot with high threshold")
    plt.axis('off')
    plt.show()

    final_boxes = nonmax_suppresion(boxes, IOU_THRESHOLD, OBJ_THRESHOLD)
    print("{} final number of boxes.".format(len(final_boxes)))
    ima = draw_box(X_test[0], final_boxes, LABELS, verbose=True)
    plt.figure(figsize=figsize)
    plt.imshow(ima)
    plt.show()

    np.random.seed(1)
    n_sample = 2
    image_names = list(np.random.choice(os.listdir(train_image_folder), n_sample))
    X_test = []
    for name in image_names:
        img_path = os.path.join(train_image_folder, name)
        _out = image_reader.fit(img_path)
        X_test.append(_out)

    X_test = np.array(X_test)
    y_pred = model.predict(X_test)
    for iframe in range(len(y_pred)):
        netout = y_pred[iframe]
        # Rescale network output
        netout_scale = output_rescaler.fit(netout)
        boxes = find_high_class_probability_bbox(netout_scale, OBJ_THRESHOLD)

        if len(boxes) > 0:
            final_boxes = nonmax_suppresion(boxes, IOU_THRESHOLD, OBJ_THRESHOLD)
            ima = draw_box(X_test[iframe], final_boxes, LABELS, verbose=True)
            plt.figure(figsize=figsize)
            plt.imshow(ima)
            plt.axis('off')
            plt.show()
