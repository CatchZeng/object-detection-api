import os
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('save_dir', None, 'workspace save dir', required=True)
flags.DEFINE_string('name', None, 'workspace name', required=True)

Dirs = [
    'annotations',
    'exported-models',
    'exported-models/ssd_mobilenet_v2_fpnlite_320x320_quant_lite',
    'exported-models/ssd_mobilenet_v2_fpnlite_640x640_quant_lite',
    'images',
    'images/test',
    'images/train',
    'images/val',
    'models',
    'models/ssd_mobilenet_v2_fpnlite_320x320',
    'models/ssd_mobilenet_v2_fpnlite_640x640',
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
