from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np
import re
import pickle as pkl
from sklearn import preprocessing
import json
import random

# load the user configs
with open('config.json') as f:
	config = json.load(f)

random.seed(config["seed"])

MAX_NB_WORDS = config["MAX_NB_WORDS"]
MAX_SEQUENCE_LENGTH=config["MAX_SEQUENCE_LENGTH"]
VALIDATION_SPLIT=config["VALIDATION_SPLIT"]
EMBEDDING_DIM=300
LSTM_OUT=config["LSTM_OUT"]
BATCH_SIZE=config["BATCH_SIZE"]
EPOCHS=config["EPOCHS"]
GLOVE_EMBEDDING_PATH=config["GLOVE_EMBEDDING_PATH"]
VECTORIZER_PATH=config["VECTORIZER_PATH"]
LABEL_ENCODER_PATH=config["LABEL_ENCODER_PATH"]
model_json_file=config["model_json_file"]
weights=config["weights"]
input_file=config["input_file"]

df = pd.read_csv(input_file,sep=",,,",header=None ,names=['question','type'])
df['type']=df['type'].str.strip()
df['question'] = df['question'].apply(lambda x: x.lower())
df['question'] = df['question'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

NUM_CLASSES=len(df['type'].unique())
print(df['type'].value_counts())

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, split=' ')
tokenizer.fit_on_texts(df['question'].values)
X = tokenizer.texts_to_sequences(df['question'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
Y = df['type']
with open(VECTORIZER_PATH, 'wb') as fil:
    pkl.dump(tokenizer, fil)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


le = preprocessing.LabelEncoder()
le.fit(Y)
Y=le.transform(Y)
labels = to_categorical(np.asarray(Y))
with open(LABEL_ENCODER_PATH, 'wb') as fil:
    pkl.dump(le, fil)

# split the data into a training set and a validation set
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * X.shape[0])

x_train = X[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = X[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

embeddings_index = {}
f = open(GLOVE_EMBEDDING_PATH, encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(LSTM_OUT, dropout_U=0.25, dropout_W=0.25))
model.add(Dense(NUM_CLASSES,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

checkpoint = ModelCheckpoint(weights, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(x_val, y_val),
          callbacks = [checkpoint,early])

# serialize model to JSON
model_json = model.to_json()
with open(model_json_file, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
# model.save_weights("model.h5")
print("Saved model to disk")
