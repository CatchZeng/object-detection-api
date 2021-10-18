import re
import os
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('dir', None, 'directory', required=True)
flags.DEFINE_string('pattern', None, 'pattern', required=True)
flags.DEFINE_string('repl', None, 'repl', required=True)


def replace(file_path, pattern, repl):
    f = open(file_path, 'r')
    alllines = f.readlines()
    f.close()
    f = open(file_path, 'w+')
    for eachline in alllines:
        a = re.sub(pattern, repl, eachline)
        f.writelines(a)
    f.close()


def get_files(path, all_files):
    file_list = os.listdir(path)
    for file in file_list:
        if ".ipynb" in file:
            print('ignore ' + file)
            continue
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            get_files(cur_path, all_files)
        else:
            p = os.path.join(path, file)
            all_files.append(p)
    return all_files


def replace_dir(dir, pattern, repl):
    files = get_files(dir, [])
    print(files)
    for f in files:
        replace(f, pattern, repl)


def main(argv):
    del argv  # Unused.
    replace_dir(FLAGS.dir, FLAGS.pattern, FLAGS.repl)


if __name__ == '__main__':
    app.run(main)