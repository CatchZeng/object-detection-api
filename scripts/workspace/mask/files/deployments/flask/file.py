import os
import uuid
from werkzeug.utils import secure_filename

# https://developer.mozilla.org/en-US/docs/Web/Media/Formats/Image_types
ALLOWED_EXTENSIONS = {
    'apng', 'avif', 'jpg', 'jpeg', 'jfif', 'pjpeg', 'pjp', 'png', 'svg',
    'webp', 'bmp', 'ico', 'cur', 'tif', 'tiff'
}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def make_upload_dir(subdir='uploads'):
    return make_dir(subdir)


def make_save_dir(subdir='static'):
    return make_dir(subdir)


def make_dir(subdir):
    basepath = os.path.dirname(__file__)
    dir = os.path.join(basepath, subdir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def file_extension(filename):
    _, extension = os.path.splitext(filename)
    return extension


def uuid_filename(filename):
    extension = file_extension(filename)
    uuid_name = uuid.uuid4().hex + extension
    uuid_name = secure_filename(uuid_name)
    return uuid_name