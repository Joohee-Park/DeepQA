# import keras
import lib.tensor as tensor
import lib.data as data
import os

from keras.layers import Input, Dense, Lambda, LSTM, RepeatVector, TimeDistributed
from keras.models import Model, Sequential
from keras import backend as K
from keras import objectives

ROOT_DIR = os.path.dirname(__file__)
# [0] Prepare answer dictionary
print("[0] Prepare Answer Dictionary")
ansToidx, idxToans = data.answer_dict()

# [1] Prepare Training Data
print("[1] Prepare Training Data")
training_sentence, training_answer = data.prepareTrainingData()

# [2] Define DeepQA Models
print("[2] Define DeepQA Models")
batch_size = 4 * 6 * 12 # To make training size divisible by batchSize
embeDim = 98
maxlen = 700
nb_epoch = 50
rnnDim = 256
denseDim = 512
answerDim = 2170

sentence = Input(batch_shape=(batch_size, maxlen, embeDim))
e1 = LSTM(rnnDim, activation='relu', return_sequences=True)(sentence)
e2 = LSTM(rnnDim, activation='relu', return_sequences=True )(e1)
e3 = LSTM(rnnDim, activation='relu')(e2)
prediction = Dense(answerDim, activation='sigmoid')(e3)

DeepQA = Model(sentence, prediction)

print(DeepQA.summary())

DeepQA.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# [3] Train the Model
print("[3] Train the Model")
DeepQA.fit(training_sentence, training_answer,
           shuffle=True,
           nb_epoch=nb_epoch,
           batch_size=batch_size)

# [4] Test
