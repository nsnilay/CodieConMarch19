from __future__ import print_function
from __future__ import absolute_import

from flask import Flask, request

import pandas as pd

from ludwig import LudwigModel

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


def get_model_definition():
    model_definition = json.loads(model_structure)
    return model_definition

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

        global ludwig_model
        print(ludwig_model.predict(dateframe))



    return "SUCCESS"



def load_model():
    print("**************Please wait....Loading weights*********************")
    global ludwig_model
    model_definition = get_model_definition()

    ludwig_model = LudwigModel(model_definition)

    train_model()

    ludwig_model.load(MODEL_PATH)
    print("weights loaded")

def train_model():
    global ludwig_model
    ludwig_model.train(data_csv='train.csv')
    ludwig_model.save(MODEL_PATH)


if __name__ == '__main__':
    print(("* Loading model and starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run(host='0.0.0.0', debug=True, threaded=False)
