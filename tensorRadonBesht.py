import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.histograms import _ravel_and_check_weights
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

#dataset
#weather data obtained from the open weather archive rp5.ru
column_names = ['datetime', 'ur/h', 'usv/h', 't', 'p0', 'humid', 'windspd', 'Td', 'rrr', 'mmprecip24', 'mmprecip48']
raw_dataset = pd.read_csv("beshtaugamma.csv", names = column_names, na_values='?', sep =',', skiprows = 1)
dataset = raw_dataset.copy()
dataset = dataset.drop('datetime', 1)
dataset = dataset.drop('ur/h', 1)
dataset = dataset.dropna()
print(dataset)

#split into test and train sets:
train_dataset = dataset.sample(frac = 0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)
print(train_dataset)

#split features from labels, label - value we're looking for
train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop('usv/h')
test_labels = test_features.pop('usv/h')

#check out the graphs
#sns.pairplot(train_dataset[['usv/h', 't', 'p0', 'humid', 'windspd', 'Td', 'rrr', 'mmprecip24', 'mmprecip48']], diag_kind='kde')
#plt.show()

#normalize values:
normalizer = preprocessing.Normalization(axis = -1)
normalizer.adapt(np.array(train_features))
first = np.array(train_features[:1])

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 0.2])
    plt.xlabel('Epoch')
    plt.ylabel('Error [t]')
    plt.legend()
    plt.grid(True)

#define the model
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
    return model

#Full DNN model
dnn_model = build_and_compile_model(normalizer)
#dnn_model.summary()
history = dnn_model.fit(
    train_features, train_labels,
    batch_size = 256,
    validation_split=0.2,
    verbose=0, epochs=5000)

plot_loss(history)
plt.show()

#Predictions
test_predictions = dnn_model.predict(test_features).flatten()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [usv/h]')
plt.ylabel('Predictions [usv/h]')
lims = [0, 1]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

#Error distribution for predictions
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [usv/h]')
_ = plt.ylabel('Count')
plt.show()
