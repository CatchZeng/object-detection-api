import os
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('save_dir', None, 'workspace save dir', required=True)
flags.DEFINE_string('name', None, 'workspace name', required=True)

Dirs = [
    'annotations',
    'exported-models',
    'exported-models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8',
    'images',
    'images/train',
    'images/val',
    'images/test',
    'images/test_annotated',
    'models',
    'models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8',
    'pre-trained-models',
]


def main(argv):
    del argv  # Unused.
    dir = os.path.join(FLAGS.save_dir, FLAGS.name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    for d in Dirs:
        p = os.path.join(dir, d)
        if not os.path.exists(p):
            os.makedirs(p)


if __name__ == '__main__':
    app.run(main)
