import glob, os
import time
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

IMAGE_DIR = './images/test/'
IMAGE_PATHS = glob.glob(os.path.join(IMAGE_DIR, '*.jp*g'))
SAVE_DIR = './images/test_annotated/'

# Load model
PATH_TO_SAVED_MODEL = './exported-models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8' + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# Labels
PATH_TO_LABELS = './annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


for image_path in IMAGE_PATHS:
    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # detection_classes should be ints.
    image_np_with_detections = image_np.copy()

    # 以下参考 https://github.com/vijaydwivedi75/Custom-Mask-RCNN_TF/blob/master/mask_rcnn_eval.ipynb
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
    detections['detection_classes'] = detections['detection_classes'][0].numpy(
    ).astype(np.uint8)
    detections['detection_boxes'] = detections['detection_boxes'][0].numpy()
    detections['detection_scores'] = detections['detection_scores'][0].numpy()
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

    plt.figure(figsize=(10, 10))
    plt.imshow(image_np_with_detections)

    (filepath, filename) = os.path.split(image_path)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path)

print('Done!')
print(f'Annotated images is at {SAVE_DIR}')