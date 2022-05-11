import tensorflow as tf
import glob
import os
import pandas as pd
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers import Dense
from keras.layers import Softmax
import matplotlib.pyplot as plt


# tweet length normalization parameter
# shorter that this will be padded with zero
# longer that this will be dropped
twl = 80

# fraction of the test data
test_fr = 0.1

# factors scale parameter, 1 - one degree precision
scale = 3

# intermediate objects
factors = {}
factors_r = {}
n = 0

# create factors
for lat in range(int(-90/scale), int((90+scale)/scale), 1):
  for lon in range(int(-180/scale), int((180+scale)/scale), 1):
    factors[(lat, lon)] = n
    n += 1


# preprocess tweet text, fill classes
def make_data(df, fname):
  seq = df.loc[:,'text']
  seq = seq.apply(lambda x: [ord(i)/int('10ffff', 16) for i in list(x)])
  seq = seq.apply(lambda x: x + (twl - len(x)) * [0])
  x = tf.constant(list(seq), dtype = tf.float32)
  x = tf.reshape(x, [x.get_shape()[0], 1, x.get_shape()[1]])
  
  lat, lon = fname[:-4].split('_')
  lat = float(lat) // scale
  lon = float(lon) // scale
  y = factors[(lat, lon)]
  y = tf.constant(len(seq) * [y], dtype = tf.float16)
  y = tf.reshape(y, [y.get_shape()[0], 1, 1])

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

# single Conv1D layer to detect phrases at once
name = 'single_layer_classification'
model = Sequential(name=name)
# Dropout layer may be added so as not to overfit, better off to test
# model.add(Dropout(.05, input_shape=(1, twl)))
model.add(Conv1D(filters=1000, kernel_size=15, strides=1, padding='same', activation='relu', input_shape=(1, twl)))
model.add(Dense((180/scale + 1) * (360/scale + 1), activation='relu'))
model.add(Softmax())

model.summary()

model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(x_train,
                    y_train,
                    batch_size=32,
                    validation_data=(x_test, y_test),
                    epochs=10,
                    verbose=2)

# plot accuracy
plt.clf()
plt.plot(history.history['val_accuracy'])
plt.title("The model's accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.tight_layout()
plt.show()

# save the model
model.save(name,
           overwrite=False,
           include_optimizer=True,
           save_format='tf')
print(f'saved as {name}')
