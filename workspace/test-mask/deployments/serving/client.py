import requests
import os
from PIL import Image
import numpy as np
import time

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as viz_utils


def workspace_path():
    depth = 3
    p = os.path.realpath(__file__)
    for _ in range(0, depth):
        p = os.path.dirname(p)
    return p


SERVER_URL = 'http://localhost:8501/v1/models/test-mask:predict'
BASE_PATH = workspace_path()
IMAGE_URL = os.path.join(BASE_PATH, 'images/test/test.jpg')
SAVE_PATH = os.path.join(BASE_PATH, 'images/test/test_annotated.jpg')

PATH_TO_LABELS = os.path.join(BASE_PATH, 'annotations/label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)


def visualize(image_np, detections):
    image_np_with_detections = image_np.copy()

    # https://github.com/vijaydwivedi75/Custom-Mask-RCNN_TF/blob/master/mask_rcnn_eval.ipynb
    # The following processing is only for single image
    detection_boxes = tf.squeeze(detections['detection_boxes'], [0])
    detection_masks = tf.squeeze(detections['detection_masks'], [0])
    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
    real_num_detection = tf.cast(detections['num_detections'][0], tf.int32)
    detection_boxes = tf.slice(detection_boxes, [0, 0],
                               [real_num_detection, -1])
    detection_masks = tf.slice(detection_masks, [0, 0, 0],
                               [real_num_detection, -1, -1])
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
    detection_masks_reframed = tf.cast(
        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    # Follow the convention by adding back the batch dimension
    detections['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

    detections['num_detections'] = int(detections['num_detections'][0])
    detections['detection_classes'] = np.array(
        detections['detection_classes'][0], dtype=np.uint8)
    detections['detection_boxes'] = np.array(detections['detection_boxes'][0])
    detections['detection_scores'] = np.array(
        detections['detection_scores'][0])
    detections['detection_masks'] = detections['detection_masks'][0].numpy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        instance_masks=detections.get('detection_masks'),
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30)

    return image_np_with_detections


def request(image_np, server_url=SERVER_URL):
    payload = {"inputs": [image_np.tolist()]}
    headers = {"content-type": "application/json"}
    start = time.time()
    res = requests.post(server_url, json=payload, headers=headers)
    end = time.time()
    print('duration: %.2fs' % (end - start))
    json = res.json()
    detections = json['outputs']
    return detections


def main():
    image = Image.open(IMAGE_URL)
    image_np = np.array(image)

    print("start request")

    detections = request(image_np)

    image_np_with_detections = visualize(image_np, detections)

    im = Image.fromarray(image_np_with_detections)
    im.save(SAVE_PATH)

    print("annotated image is saved to " + SAVE_PATH)


if __name__ == '__main__':
    main()
