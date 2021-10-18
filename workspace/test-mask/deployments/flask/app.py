from flask import Flask, request, url_for
import os
from api import response_data, response_err, Error
from file import allowed_file, make_upload_dir, uuid_filename
from model import predict_image

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['UPLOAD_FOLDER'] = make_upload_dir()

SERVER_URL = os.getenv('SERVER_URL')
print('SERVER_URL:' + SERVER_URL)


@app.route('/test-mask/v1/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return response_err(Error.NO_FILE_PART)

    f = request.files['image']
    if f.filename == '':
        return response_err(Error.NO_SELECTED_FILE)

    if not allowed_file(f.filename):
        return response_err(Error.NO_ALLOWED_FILE)

    filename = uuid_filename(f.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(file_path)

    predict_image(SERVER_URL, file_path, filename)

    url = url_for('static', filename=filename)

    return response_data('predict successfully', {
        "image_url": url,
    })


@app.route("/")
def hello_world():
    return "It works!"