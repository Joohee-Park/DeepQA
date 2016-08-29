# import keras
import lib.tensor as tensor
import lib.data as data
import os

from keras.layers import Input, Dense, Lambda, LSTM, RepeatVector, TimeDistributed
from keras.models import Model, Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras import backend as K
from keras import objectives

import pickle


ROOT_DIR = os.path.dirname(__file__)
# [0] Prepare answer dictionary
print("[0] Prepare Answer Dictionary")
ansToidx, idxToans = data.answer_dict()

# [1] Prepare Training Data
print("[1] Prepare Training Data")
training_sentence, training_answer = data.prepareTrainingData()
print("[1] Number of training instances is " + str(training_answer.shape[0]))

# [2] Define DeepQA Models
print("[2] Define DeepQA Models")
batch_size = 4 * 6 * 12 # To make training size divisible by batchSize
embeDim = 98
maxlen = 500
nb_epoch = 50
rnnDim = 128
answerDim = 150

MODEL_PATH = "Model/model.h5"
if os.path.exists(MODEL_PATH):
    print("[2] Trained model already exist. Load the existing model")
    DeepQA = load_model(MODEL_PATH)
else:
    print("[2] Trained model not found. Start to build a fresh model")

    sentence = Input(batch_shape=(batch_size, maxlen, embeDim))
    e1 = LSTM(rnnDim, activation='relu', return_sequences=True)(sentence)
    e2 = LSTM(rnnDim, activation='relu')(e1)
    prediction = Dense(answerDim, activation='sigmoid')(e2)

    DeepQA = Model(sentence, prediction)

    print(DeepQA.summary())

    DeepQA.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# [3] Train the Model
print("[3] Train the Model")
DeepQA.fit(training_sentence, training_answer,
           shuffle=True,
           nb_epoch=nb_epoch,
           batch_size=batch_size)

if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)
DeepQA.save(MODEL_PATH)
print("[3] Successfully save the Model")

# [4] Test
