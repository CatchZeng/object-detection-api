import os
import json
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('train_json', 'images/train.json', 'train.json file path')
flags.DEFINE_string('save_dir', 'annotations', 'save directory')


def convert_classes(classes, start=1):
    msg = ''
    for id, name in enumerate(classes, start=start):
        msg = msg + "item {\n"
        msg = msg + "  id: " + str(id) + "\n"
        msg = msg + "  name: '" + name + "'\n}\n\n"
    return msg[:-1]


def main(argv):
    del argv  # Unused.

    classes = []
    with open(FLAGS.train_json, 'r') as load_f:
        dict = json.load(load_f)
        categories = dict['categories']
        for c in categories:
            classes.append(c['name'])

    txt = convert_classes(classes)
    print(txt)

    save_path = os.path.join(FLAGS.save_dir, 'label_map.pbtxt')
    with open(save_path, 'w') as dump_f:
        dump_f.write(txt)


if __name__ == '__main__':
    app.run(main)