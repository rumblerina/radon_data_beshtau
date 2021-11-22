import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib.dates import DateFormatter
import numpy as np
from numpy.core.arrayprint import DatetimeFormat
from numpy.lib.histograms import _ravel_and_check_weights
import pandas as pd
from pandas.io.formats.format import Datetime64Formatter
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import warnings
import datetime as dt
warnings.simplefilter(action='ignore', category=FutureWarning)
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#dataset
column_names = ['datetime', 'ur/h', 'usv/h', 't', 'p0', 'humid', 'windspd', 'Td', 'rrr', 'mmprecip24', 'mmprecip48']
raw_dataset = pd.read_csv("beshtaugamma.csv", names = column_names, na_values='?', sep =',', skiprows = 1)
dataset = raw_dataset.copy()
dataset = dataset.drop('ur/h', 1)
dataset = dataset.dropna()
timestamps_orig = dataset.pop('datetime')
#print(pred_set)

#split into test and train sets:
train_dataset = dataset.sample(frac = 0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)
print(train_dataset)
print(test_dataset)

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
        layers.Dense(1)
    ])
    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
    return model

callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)

#Full DNN model
dnn_model = build_and_compile_model(normalizer)
#dnn_model.summary()
history = dnn_model.fit(
    train_features, train_labels,
    batch_size = 256,
    validation_split=0.2,
    verbose=1, epochs=1000,
    callbacks = [callback])

# plot_loss(history)
# plt.show()

#Predictions
test_predictions = dnn_model.predict(test_features).flatten()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [usv/h]')
plt.ylabel('Predictions [usv/h]')
plt.show()

#Error distribution for predictions
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [usv/h]')
_ = plt.ylabel('Count')
plt.show()

#Show original plot
y_orig = dataset['usv/h']
plt.plot(timestamps_orig,y_orig)
plt.gcf().autofmt_xdate()
plt.xticks(timestamps_orig[::56])
plt.xlabel("Date")
plt.ylabel("Dose rate, usv/h")
plt.title("Actual doserate at mt. Beshtau in 2018-19")
plt.show()
 
#define the savitzky-golay smoothing algorithm
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as np
    from math import factorial
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError('window_size and order have to be type int')
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError('Window size must be a positive odd number')
    if window_size < order + 2:
        raise TypeError('Window_size is too small for the polynomials order')
    order_range = range(order+1)
    half_window = (window_size - 1) // 2
    #precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode = 'valid')

#Make predictions for given data
p_column_names = ['datetime', 't', 'p0', 'humid', 'windspd', 'Td', 'rrr', 'mmprecip24', 'mmprecip48']
ds_pred = pd.read_csv("beshtaugamma2020.csv", names = p_column_names, na_values='?', sep =',', skiprows = 1)
timestamps = ds_pred.pop('datetime')
predix = dnn_model.predict(ds_pred).flatten()
timestamps = timestamps[:-1]
pred_smoothed = savitzky_golay(predix, 25, 2)
predix = predix[:-1]
#This is the unsmoothed plot
#plt.plot(timestamps,predix, color = 'red')
#This is the smoothed plot
plt.plot(timestamps,pred_smoothed, color = 'blue')
plt.gcf().autofmt_xdate()
plt.xticks(timestamps[::56])
plt.xlabel("Date")
plt.ylabel("Dose rate, usv/h")
plt.title('Predicted dose rate at mt. Beshtau in 2019-20')
plt.show()

#sns.pairplot(raw_dataset, kind = 'reg', plot_kws = {'line_kws':{'color':'blue'}})
#plt.show()
