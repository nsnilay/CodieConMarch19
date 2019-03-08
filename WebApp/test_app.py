from __future__ import print_function
from __future__ import absolute_import

from flask import Flask, request

from dateparser.search import search_dates

import pandas as pd

# from ludwig import LudwigModel

import json

model_structure = """{
	"input_features": [
		{
			"name": "utterance",
			"type": "sequence",
			"encoder": "rnn",
			"cell_type": "lstm",
			"bidirectional": true,
			"num_layers": 2,
			"reduce_output": null
		}
	],
	"output_features": [
		{
			"name": "intent",
			"type": "category",
			"reduce_input": "sum",
			"num_fc_layers": 1,
			"fc_size": 64
		}
	]
}"""


# def get_model_definition():
#     model_definition = json.loads(model_structure)
#     return model_definition

# from WebApp.models.model import get_model_definition

app = Flask(__name__)

ludwig_model = None

MODEL_PATH = "myTestPrediction"

@app.route("/")
def hello():
    return "Welcome to DNA"



@app.route("/predict", methods = ['GET'])
def predict():

    if request.method == 'GET':
        data = request.data
        print(data)
        request_body = json.loads(data)
        print(request_body)
        print(request_body["text"])
        dateframe = pd.DataFrame([request_body["text"]], columns=['utterance'])
        print(dateframe)
        extract_dates(request_body["text"])

    return "SUCCESS"

def extract_dates(sentence):
    """
    :param sentence: the sentence from which the dates have to be extracted
    :return: the list of dates extracted
    """
    return date_wrapper(sentence)



def date_wrapper(sentence):
    """
    @method is our own wrapper to do any preprocessing on the sentence before sending
    it to python's dateparser library
    :param sentence:
    :return: the list of dates extracted
    """

    list_of_dates = search_dates(sentence)
    for date in list_of_dates:
        print(date[1])

if __name__ == '__main__':
    print(("* Loading model and starting server..."
        "please wait until server has fully started"))
    app.run(host='0.0.0.0', port=6969, debug=True, threaded=False)
