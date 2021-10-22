from flask import Flask, request, url_for
from flask_cors import CORS
import os
from api import response_data, response_err, Error
from file import allowed_file, make_upload_dir, uuid_filename
from model import predict_image

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False
app.config['UPLOAD_FOLDER'] = make_upload_dir()

SERVER_URL_ENV = os.getenv('SERVER_URL')
SERVER_URL = SERVER_URL_ENV if SERVER_URL_ENV else "http://localhost:8501/v1/models/{{.Name}}:predict"
print('SERVER_URL:' + SERVER_URL)


@app.route('/{{.Name}}/v1/predict', methods=['POST'])
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