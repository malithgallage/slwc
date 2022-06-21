"""
ara 58tama plot karl balamu
Csv ekkata wage
Csv ekkata loss,accuracy, file name, white pixel count, black pixek count
--> After pixel counts tika da plot kale?? Kalin ewa mecchara adu wenna puluwanda?? max eka tyenneth 385....
--> After pixel newe pre count eka thamai one...Or dekama plot karala balamu..me tibba output walin matanm idea ekk ganna be...
--> 0 nathuwa 63 thibba ewath count karada white list ekata??? mechchara adu wenna be white pixel count eka....
"""
import os
from model import *
from data import *
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import History
from skimage import io
import skimage.transform as trans
import tensorflow as tf
import segmentation_models as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from tensorflow.keras import backend as K
from mode.config import *
#import segmentation_models
from segmentation_models.metrics import iou_score, f1_score
from segmentation_models.losses import jaccard_loss, dice_loss, binary_focal_loss, BinaryFocalLoss
from glob import glob

arg = command_arguments()

train_path = arg.train_path
train_img_folder = arg.train_img_folder
train_label_folder = arg.train_label_folder
img_size = (256,256)

model_name = arg.model_name

# inference
print(model_name)
model = load_model(model_name, custom_objects={'iou_score':iou_score, 'f1-score':f1_score})
# model.summary()
imagepaths = sorted(glob(os.path.join(train_path, train_img_folder, 'sperm', 'image_*')))
labelpaths = sorted(glob(os.path.join(train_path, train_label_folder, 'sperm', 'image_*')))

# metrics
accuracies = []
binary_accuracies = []
iou_scores = []
f1_scores = []

# losses
bces = []
jaccard_losses = []
dice_losses = []
binary_focal_losses = []
binary_focal_losses3 = []

# other stats
black_pixel_counts = []
white_pixel_counts = []

for imagepath, labelpath in zip(imagepaths, labelpaths):
  print(imagepath)
  x = io.imread(imagepath, as_gray=True)
  y_true = io.imread(labelpath, as_gray=True)
  x = trans.resize(x, img_size)
  x = x.reshape(*img_size, 1)
  x = np.expand_dims(x, axis=0)
  y_true = trans.resize(y_true, img_size)
  x, y_true = adjustData(x,y_true)
  y_pred = model.predict(x)
  y_pred = y_pred.reshape(*img_size)

  m = tf.keras.metrics.Accuracy()
  m.update_state(y_true, y_pred)
  accuracy = m.result().numpy()
  accuracies.append(accuracy)
  print(accuracy)

  m = tf.keras.metrics.BinaryAccuracy()
  m.update_state(y_true, y_pred)
  binary_accuracy = m.result().numpy()
  binary_accuracies.append(binary_accuracy)
  print(binary_accuracy)

  # m = sm.metrics.IOUScore()
  # m.update_state(y_true, y_pred)
  # ious = m.result().numpy()
  # iou_scores.append(ious)
  iou_score = sm.metrics.IOUScore()
  y_true_ = np.expand_dims(y_true,axis=0)
  y_pred_ = np.expand_dims(y_pred,axis=0)
  ious = iou_score(y_true_, y_pred_).numpy()
  iou_scores.append(ious)
  print(ious)

  f1_score = sm.metrics.FScore()
  f1s = f1_score(y_true_.astype('float64'), y_pred_.astype('float64')).numpy()
  f1_scores.append(f1s)
  print(f1s)

  bcef = tf.keras.losses.BinaryCrossentropy()
  bce  = bcef(y_true,y_pred).numpy()
  bces.append(bce)
  print(bce)

  jl  = jaccard_loss(y_true_,y_pred_).numpy()
  jaccard_losses.append(jl)
  print(jl)

  dl  = dice_loss(y_true_.astype('float64'),y_pred_.astype('float64')).numpy()
  dice_losses.append(dl)
  print(dl)

  bfl  = binary_focal_loss(K.constant(y_true_.astype('float64')),K.constant(y_pred_.astype('float64'))).numpy()
  binary_focal_losses.append(bfl)
  print(bfl)

  binary_focal_loss3 = BinaryFocalLoss(gamma=3.0)
  bfl3  = binary_focal_loss3(K.constant(y_true_.astype('float64')),K.constant(y_pred_.astype('float64'))).numpy()
  binary_focal_losses3.append(bfl3)
  print(bfl3)

  black_pixel_count = (y_true==0).sum()
  black_pixel_counts.append(black_pixel_count)
  white_pixel_count = (y_true==1).sum()
  white_pixel_counts.append(white_pixel_count)
  assert white_pixel_count+black_pixel_count==y_true.ravel().shape[0]

image_names = [os.path.basename(imagepath) for imagepath in imagepaths]

pd.DataFrame({'file_name':image_names, 
              'accuracy':accuracies,
              'binary_accuracy':binary_accuracies,
              'iou_score':iou_scores,
              'f1_score':f1_scores,
              'bce':bces,
              'jaccard_loss':jaccard_losses,
              'dice_loss':dice_losses,
              'binary_focal_loss':binary_focal_losses,
              'binary_focal_loss3':binary_focal_losses3,
              'black_pixel_count':black_pixel_counts, 
              'white_pixel_count':white_pixel_counts}).to_csv('model_check_%s.csv'%model_name, index=False)

