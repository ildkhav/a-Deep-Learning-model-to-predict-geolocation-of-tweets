import sys
import tensorflow as tf
import numpy as np
import re
import matplotlib.pyplot as plt


try:
  model_name = sys.argv[1]
  text = sys.argv[2]
  twl = int(sys.argv[3])
except:
  print("Usage: predict.py <model name> <text> <model's max tweet length> [the scale value for classification model]")
  exit()


def MeanErrorKm(y_true, y_pred):
  return None

c = [ord(i)/int('10ffff', 16) for i in list(text)] + (twl - len(text)) * [0]
x = tf.constant(c, dtype=tf.float32)
x = tf.reshape(x, [1, 1, tf.shape(x).numpy()[0]])

try:
  if re.search('regression$', model_name):
    model = tf.keras.models.load_model(model_name, custom_objects={'MeanErrorKm': MeanErrorKm})
    m = np.ndarray((1,1,2), buffer=np.array([180, 360], dtype=float))
    s = np.ndarray((1,1,2), buffer=np.array([90, 180], dtype=float))
    print(model.predict(x) * m - s)

  elif re.search('classification$', model_name):
    try:
      scale = int(sys.argv[4])
    except Exception:
      print("Usage: predict.py <model name> <text> <model's max tweet length> [the scale value for classification model]")
      exit()

    factors_r = {}
    points_l = []
    n = 0
    for lat in range(int(-90/scale), int((90+scale)/scale), 1):
      for lon in range(int(-180/scale), int((180+scale)/scale), 1):
        factors_r[n] = [lat*scale, lon*scale]
        points_l.append([lat*scale, lon*scale])
        n += 1

    points_l = tf.transpose(tf.constant(points_l))

    model = tf.keras.models.load_model(model_name)
    # c = factors_r[np.argmax(model.predict(x))]
    print(factors_r[np.argmax(model.predict(x))])
    # print(tf.constant(c, dtype=tf.float32) * scale)
    
    pr = model.predict(x)
    plt.clf()
    plt.scatter(points_l[1], points_l[0], s = [1] * len(pr[0][0]), c = 1 / pr[0][0], cmap='gray')
    plt.title("Coordinates confidense distribution")
    plt.ylabel('lattitude')
    plt.xlabel('longtitude')
    plt.grid()
    plt.tight_layout()
    plt.show()
  else:
    print('unexpected file name')
    exit()
    
except Exception:
  print(sys.exc_info())
  exit()
