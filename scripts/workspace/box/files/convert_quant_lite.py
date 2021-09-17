from absl import app
from absl import flags
import tensorflow as tf
import os

FLAGS = flags.FLAGS

flags.DEFINE_string('saved_model_dir', None, 'Path to a save_model')
flags.DEFINE_string('output_file', None, 'Path to write output file.')

flags.mark_flag_as_required('saved_model_dir')
flags.mark_flag_as_required('output_file')


def main(_):
    converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    with open(FLAGS.output_file, 'wb') as f:
        f.write(tflite_quant_model)


if __name__ == '__main__':
    app.run(main)
