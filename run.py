import numpy as np
import keras
import pickle as pkl
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import json
import random
import argparse
import re

parser = argparse.ArgumentParser(description='Input the question')
parser.add_argument('-q','--question', help='Enter your Question here',
                    default='What time does the train leave')
args = vars(parser.parse_args())

# load the user configs
with open('config.json') as f:
	config = json.load(f)

random.seed(config["seed"])

MAX_SEQUENCE_LENGTH=config["MAX_SEQUENCE_LENGTH"]
VECTORIZER_PATH=config["VECTORIZER_PATH"]
LABEL_ENCODER_PATH=config["LABEL_ENCODER_PATH"]
model_json_file=config["model_json_file"]
weights=config["weights"]

print("Load questions :")
raw_text=args['question']
test_list=[re.sub('[^a-zA-z0-9\s]','',raw_text.lower())]
print(test_list)

with open(LABEL_ENCODER_PATH, "rb") as fil:
    le = pkl.load(fil)

with open(VECTORIZER_PATH, "rb") as fil:
    tokenizer = pkl.load(fil)

example = tokenizer.texts_to_sequences(test_list)
example = pad_sequences(example, maxlen=MAX_SEQUENCE_LENGTH)

# load json and create model
json_file = open(model_json_file, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(weights)
print("Loaded model from disk")

print("Question Type :",le.inverse_transform(np.argmax(model.predict(example))))
