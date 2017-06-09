"""
Python3: Deep learning applied to sklearn small dataset


To run first install Keras
pip3 install keras

And then install Tensorflow or Theano.

To run do
python3 main.py

"""

import numpy as np
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense

EPOCHS = 3000
BATCH_SIZE = 100

iris = datasets.load_iris()

# The inputs
print(iris.data)
x = iris.data

# The outputs: multi-class classification.
print(iris.target)

y = iris.target


# Construct a model
model = Sequential()

layer = Dense(4, input_dim=4, init='glorot_normal', activation='relu')
model.add(layer)

layer2 = Dense(3, init='glorot_normal', activation='sigmoid')
model.add(layer2)

layer3 = Dense(1, init='glorot_normal', activation='linear')
model.add(layer3)

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, nb_epoch=EPOCHS, batch_size=BATCH_SIZE)


def accuracy(y, y_pred):
    return sum([e_y == e_ypred for e_y, e_ypred in zip(y, y_pred)])/len(y)

# prediction:
y_pred = np.array([int(round(e[0])) for e in model.predict(x)])
print(y_pred)

# real data:
print(y)

print('accuracy: ' + str(accuracy(y, y_pred)))
