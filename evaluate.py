import sys
import tensorflow as tf
import glob
import pandas as pd
import re
import os
import math


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


def make_data(df, fname):
  seq = df.loc[:,'text']
  seq = seq.apply(lambda x: [ord(i)/int('10ffff', 16) for i in list(x)])
  seq = seq.apply(lambda x: x + (twl - len(x)) * [0])
  x = tf.constant(list(seq), dtype = tf.float32)
  x = tf.reshape(x, [x.get_shape()[0], 1, x.get_shape()[1]])
  
  lat, lon = fname[:-4].split('_')
  
  if re.search('regression$', model_name):
    y = tf.constant(len(seq) * [[(float(lat) + 90)/180, (float(lon) + 180)/360]], dtype = tf.float32)
    y = tf.reshape(y, [y.get_shape()[0], 1, y.get_shape()[1]])
  else:
    lat = float(lat) // scale
    lon = float(lon) // scale
    y = factors[(lat, lon)]
    y = tf.constant(len(seq) * [y], dtype = tf.float16)
    y = tf.reshape(y, [y.get_shape()[0], 1, 1])

  return([x, y])


try:
  model_name = sys.argv[1]
  twl = int(sys.argv[2])
except:
  print("Usage: evaluate.py <model name> <model's max tweet length> [the scale value for classification model]")
  exit()

try:
  if re.search('regression$', model_name):
    model = tf.keras.models.load_model(model_name, custom_objects={'MeanErrorKm': MeanErrorKm})
  elif re.search('classification$', model_name):
    try:
      scale = int(sys.argv[3])
    except Exception:
      print("Usage: evaluate.py <model name> <model's max tweet length> [the scale value for classification model]")
      exit()

    factors = {}
    n = 0
    for lat in range(int(-90/scale), int((90+scale)/scale), 1):
      for lon in range(int(-180/scale), int((180+scale)/scale), 1):
        factors[(lat, lon)] = n
        n += 1

    model = tf.keras.models.load_model(model_name)
  else:
    print('unexpected file name')
    exit()
    
except Exception:
  print(sys.exc_info())
  exit()

for f in glob.glob('./*csv'):
  df = pd.read_csv(f, sep=';')

  if df.shape[0] == 0:
    continue

  df = df[df['text'].map(len) <= twl]

  x, y = make_data(df, os.path.basename(f))

if re.search('regression$', model_name):
  loss, root_mean_squared_error, mean_absolute_error, MeanErrorKm = model.evaluate(x, y, verbose=0)
  print(f'mean error {round(MeanErrorKm)} km')
else:
  loss, accuracy = model.evaluate(x, y, verbose=0)
  print(f'accuracy {accuracy}')

