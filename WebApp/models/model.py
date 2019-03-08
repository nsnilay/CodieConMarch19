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
