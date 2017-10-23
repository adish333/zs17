from timeit import default_timer as timer
ti = timer()
print("started at: ", ti)

import os
import warnings
import pandas as pd
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import TimeDistributed
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")  # Hide messy Numpy warnings

def doWork(train, user):
    SEQ_LENGTH = 5
    HIDDEN_DIM = 30
    LAYER_NUM = 2
    nb_epoch = 250
    BATCH_SIZE = 1
    GENERATE_LENGTH = 10

    df = train[train['id'] == user]
    data = np.array(df['event'])
    chars = list(set(data))
    VOCAB_SIZE = len(chars)

    char_to_ix = {char: ix for ix, char in enumerate(chars)}
    X = np.zeros((len(data) / SEQ_LENGTH, SEQ_LENGTH, VOCAB_SIZE))
    y = np.zeros((len(data) / SEQ_LENGTH, SEQ_LENGTH, VOCAB_SIZE))
    for i in range(0, int(len(data) / SEQ_LENGTH)):
        X_sequence = data[i * SEQ_LENGTH:(i + 1) * SEQ_LENGTH]
        X_sequence_ix = [char_to_ix[value] for value in X_sequence]
        input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
        for j in range(SEQ_LENGTH):
            input_sequence[j][X_sequence_ix[j]] = 1.
        X[i] = input_sequence

        y_sequence = data[i * SEQ_LENGTH + 1:(i + 1) * SEQ_LENGTH + 1]
        y_sequence_ix = [char_to_ix[value] for value in y_sequence]
        target_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
        for j in range(SEQ_LENGTH):
            target_sequence[j][y_sequence_ix[j]] = 1.
        y[i] = target_sequence

    model = Sequential()
    model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
    for j in range(LAYER_NUM - 1):
        model.add(LSTM(HIDDEN_DIM, return_sequences=True))
        model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(VOCAB_SIZE)))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, nb_epoch=nb_epoch)
    model.save_weights('user_{}_checkpoint_{}_epoch_{}.hdf5'.format(i, HIDDEN_DIM, nb_epoch))
    output = generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, chars)
    return output


def generate_text(model, length, VOCAB_SIZE, chars):
    ix_to_char = {ix: char for ix, char in enumerate(chars)}
    ix = [np.random.randint(VOCAB_SIZE)]
    y_char = [ix_to_char[ix[-1]]]
    X = np.zeros((1, length, VOCAB_SIZE))
    for i in range(length):
        X[0, i, :][ix[-1]] = 1
        print(ix_to_char[ix[-1]], end=" ")
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return y_char


train=pd.read_csv("data.csv")
train['event']= train['event'].astype("category")
userdata=pd.read_csv("user.csv")
users = userdata['PID']
not_done=[]
submission = pd.DataFrame()

x=0
for user in users:
    x=x+1
    print("\ncalculating for user:",user)
    try:
        out = doWork(train, user)
        submission[user]=out
        file='{}.json'.format(user)
        with open(file, 'w') as outfile:
            json.dump(out, outfile)
    except:
        not_done.append(user)
        continue
    if(x==50):
        f1='notdone{}.csv'.format(user)
        f2='submission{}.csv'.format(user)
        remaining = pd.DataFrame()
        remaining['notdone'] = not_done
        remaining.to_csv(f1)
        submission.to_csv(f2)
        x=0     
        
remaining = pd.DataFrame()
remaining['notdone'] = not_done
remaining.to_csv('notdone.csv')
submission.to_csv('submission.csv')
tf = timer()
print("time taken: ",tf - ti)
