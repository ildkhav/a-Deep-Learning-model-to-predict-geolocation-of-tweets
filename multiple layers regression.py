import math
import tensorflow as tf
import glob
import pandas as pd
import os
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers import Dense
import matplotlib.pyplot as plt


# tweet length normalization parameter
# shorter that this will be padded with zero
# longer that this will be dropped
twl = 80

# fraction of the test data
test_fr = 0.1


# a metric to compute model's error in kilometers
def MeanErrorKm(y_true, y_pred):
  m = tf.constant([180, 360], dtype=tf.float32)
  s = tf.constant([90, 180], dtype=tf.float32)

  y_true = y_true * m - s
  y_pred = y_pred * m - s 
  
  y_true = y_true * math.pi / 180
  y_pred = y_pred * math.pi / 180

  y_true = tf.transpose(y_true)
  y_pred = tf.transpose(y_pred)

  dlat = y_pred[0] - y_true[0]
  dlon = y_pred[1] - y_true[1]

  m = tf.math.sin(dlat / 2)**2 + tf.math.cos(y_true[0]) * tf.math.cos(y_pred[0]) * tf.math.sin(dlon / 2)**2

  return(tf.reduce_mean(2 * 6371 * tf.math.asin(tf.math.sqrt(m))))


# preprocess tweet text and get coordinates
def make_data(df, fname):
  seq = df.loc[:,'text']
  seq = seq.apply(lambda x: [ord(i)/int('10ffff', 16) for i in list(x)])
  seq = seq.apply(lambda x: x + (twl - len(x)) * [0])
  x = tf.constant(list(seq), dtype = tf.float32)
  x = tf.reshape(x, [x.get_shape()[0], 1, x.get_shape()[1]])

  # taking coordinates from the file name
  lat, lon = fname[:-4].split('_')
  y = tf.constant(len(seq) * [[(float(lat) + 90)/180, (float(lon) + 180)/360]], dtype = tf.float32)
  y = tf.reshape(y, [y.get_shape()[0], 1, y.get_shape()[1]])

  return([x, y])


# intermediate lists
x_test_l = []
y_test_l = []

x_train_l = []
y_train_l = []

# a loop to read data
for f in glob.glob('./*csv'):

  df = pd.read_csv(f, sep=';')

  if df.shape[0] == 0:
    continue

  df = df[df['text'].map(len) <= twl]

  df_test = df.sample(frac = test_fr)

  # in case there are too few rows to get fraction, simply take the first row
  if df_test.shape[0] == 0:
    df_test = df.iloc[0,:]
    df_test = df_test.to_frame()
    df_test = df_test.transpose()
    df_train = df.iloc[1:,:]
  else:
    df_train = df.drop(df_test.index)

  x, y = make_data(df_test, os.path.basename(f))
  x_test_l.append(x)
  y_test_l.append(y)

  x, y = make_data(df_train, os.path.basename(f))
  x_train_l.append(x)
  y_train_l.append(y)

# the train, test data to work with
x_train = tf.concat(x_train_l, 0)
y_train = tf.concat(y_train_l, 0)
x_test = tf.concat(x_test_l, 0)
y_test = tf.concat(y_test_l, 0)

# multiple Conv1D layers to detect syllables up to phrases
name = 'multiple_layers_regression'
model = Sequential(name=name)
# Dropout layer may be added so as not to overfit, better off to test
# model.add(Dropout(.05, input_shape=(1, twl)))
model.add(Conv1D(filters=1000, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(1, twl)))
model.add(Conv1D(filters=1000, kernel_size=2, strides=1, padding='same', activation='relu'))
model.add(Conv1D(filters=1000, kernel_size=2, strides=1, padding='same', activation='relu'))
model.add(Conv1D(filters=1000, kernel_size=2, strides=1, padding='same', activation='relu'))
model.add(Dense(2, activation='relu'))

model.summary()
                 
model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.RootMeanSquaredError(),
                       tf.keras.metrics.MeanAbsoluteError(),
                       MeanErrorKm])

history = model.fit(x_train,
                    y_train,
                    batch_size=32,
                    validation_data=(x_test, y_test),
                    # 30 were enough, adjust if needed
                    epochs=10,
                    verbose=2)

# plot RMSE, MAE, normalized degree alues
plt.clf()
plt.plot(history.history['val_root_mean_squared_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title("Validation error, normalized degrees")
plt.ylabel('error value')
plt.xlabel('epoch')
plt.legend(['RMSE', 'MAE'], loc='upper left')
plt.tight_layout()
plt.show()

# plot MAE in km
plt.clf()
plt.plot(history.history['val_MeanErrorKm'])
plt.title("Validation error in km")
plt.ylabel('km')
plt.xlabel('epoch')
plt.tight_layout()
plt.show()

# save the model
model.save(name,
           overwrite=False,
           include_optimizer=True,
           save_format='tf')
print(f'saved as {name}')
