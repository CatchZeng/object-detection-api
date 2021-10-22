from enum import Enum
import json


class Response:
    def __init__(self, code, message, data=''):
        self.code = code
        self.data = data
        self.message = message


class Error(Enum):
    NO_FILE_PART = Response(1000, "No image file part")
    NO_SELECTED_FILE = Response(1001, "No selected file")
    NO_ALLOWED_FILE = Response(1002, "No allowed file")


def response_data(message, data=''):
    res = Response(0, message, data)
    return jsonify(res), 200, {
        "Content-Type": "application/json; charset=utf-8"
    }


def response_err(e):
    res = e.value
    return jsonify(res), 200, {
        "Content-Type": "application/json; charset=utf-8"
    }


def jsonify(r):
    return json.dumps(r.__dict__, ensure_ascii=False)
