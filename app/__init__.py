from flask import Flask
from flask_cors import CORS
from pathlib import Path

PATH_TO_APP = Path(__file__).absolute().parent.resolve()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

__all__ = ["app", "PATH_TO_APP"]
