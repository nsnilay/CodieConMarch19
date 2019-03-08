from ludwig import LudwigModel
import json


MODEL_PATH = "myTestPrediction"

data = """{
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

model_definition = json.loads(data)

ludwig_model = LudwigModel(model_definition)
loaded_ludwig_model = ludwig_model
train_stats = ludwig_model.train(data_csv='train.csv')

ludwig_model.save(MODEL_PATH)
loaded_ludwig_model.load(MODEL_PATH)
predictions = loaded_ludwig_model.predict(data_csv='test.csv')

print(predictions)
