import os
import glob
from absl import app
from absl import flags
import xml.etree.ElementTree as ET

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'xml_dir', 'images/train',
    'Path to the folder where the input .xml files are stored.')
flags.DEFINE_string('save_dir', 'annotations', 'save directory')


def convert_classes(classes, start=1):
    msg = ''
    for id, name in enumerate(classes, start=start):
        msg = msg + "item {\n"
        msg = msg + "  id: " + str(id) + "\n"
        msg = msg + "  name: '" + name + "'\n}\n\n"
    return msg[:-1]


def get_classes(path):
    classes = []

    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            class_name = member.find('name').text
            if class_name not in classes:
                classes.append(class_name)

    print("unsorted: ", classes)
    classes.sort()
    print("sorted: ", classes)
    return classes


def main(argv):
    del argv  # Unused.
    classes = get_classes(FLAGS.xml_dir)
    txt = convert_classes(classes)
    print(txt)

    save_path = os.path.join(FLAGS.save_dir, 'label_map.pbtxt')
    with open(save_path, 'w') as dump_f:
        dump_f.write(txt)


if __name__ == '__main__':
    app.run(main)