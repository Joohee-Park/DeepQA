# import keras
import lib.tensor as tensor
import lib.data as data
import os

from keras.layers import Input, Dense, LSTM
from keras.models import Model, load_model
import numpy as np

embeDim = 98
maxlen = 500
nb_epoch = 50
rnnDim = 128
answerDim = 50
batch_size = 100

ROOT_DIR = os.path.dirname(__file__)
# [0] Prepare answer dictionary
print("[0] Prepare Answer Dictionary")
ansToidx, idxToans = data.answer_dict()

# [1] Prepare Training Data
print("[1] Prepare Training Data")
_training_sentence, _training_answer = data.prepareTrainingData()

# [1.1] Cut the residual training data to fit batch size
training_size = _training_answer.shape[0]
training_size -= (training_size % batch_size)

training_sentence = _training_sentence[0:training_size,:,:]
training_answer = _training_answer[0:training_size,:]

print("[1] Number of training instances is " + str(training_size))

print("[1] Training Label sanity check : " , end="")

if np.sum(np.sum(training_answer)) == training_size:
    print("PASSED")
else:
    print("FAILED")
    exit()

# [2] Define DeepQA Models
print("[2] Define DeepQA Models")

MODEL_PATH = "Model/model.h5"
if os.path.exists(MODEL_PATH):
    print("[2] Trained model already exist. Load the existing model")
    DeepQA = load_model(MODEL_PATH)
else:
    print("[2] Trained model not found. Start to build a fresh model")

    sentence = Input(batch_shape=(batch_size, maxlen, embeDim))
    e1 = LSTM(rnnDim, activation='tanh', return_sequences=True)(sentence)
    e2 = LSTM(rnnDim, activation='tanh')(e1)
    prediction = Dense(answerDim, activation='softmax')(e2)

    DeepQA = Model(sentence, prediction)

    print(DeepQA.summary())

    DeepQA.compile(optimizer='adadelta', loss='rms', metrics=['accuracy'])

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
