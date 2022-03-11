import json
import logging

from flask import Flask

from logger import set_logger

# read artifacts file
with open("artifacts.json", "r") as f:
    artifacts = json.load(f)
    log = artifacts['logging']

# set logger
logger = logging.getLogger(__name__)
logger = set_logger(logger, log['dir'], log['files']['main'])

# flask app
app = Flask(__name__)


@app.route('/')
def home():
    return 'getting started'


@app.route('/ingestion')
def data_ingestion():
    return "this is data injection route"


if __name__ == '__main__':
    app.run(debug=True)
