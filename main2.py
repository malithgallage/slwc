import os
import os.path
from model import *
from data import *
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import History
import tensorflow as tf
import matplotlib.pyplot as plt 
from tensorflow.keras import backend as K
from mode.config import *
from csvrecord import * 
from pathlib import Path
from custom_losses import f1_loss
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import segmentation_models as sm


arg = command_arguments()

batch_size = arg.batchsize
train_path = arg.train_path
train_img_folder = arg.train_img_folder
train_label_folder = arg.train_label_folder
val_path = arg.val_path
val_img_folder = arg.val_img_folder
val_label_folder = arg.val_label_folder
test_img_path = arg.test_img_path
steps_per_epoch = arg.steps_per_epoch
epochs = arg.epochs
save_result_folder = arg.save_result_folder
csvfilename = arg.csvfilename
model_name = arg.model_name
plt_save_name = arg.plt_save_name
val_plt_name = arg.val_plt_name
img_num = arg.img_num
filenum = arg.filenum

#augs 

rotation_range = arg.rotation_range
width_shift_range = arg.width_shift_range
height_shift_range = arg.height_shift_range
zoom_range = arg.zoom_range
horizontal_flip = arg.horizontal_flip
fill_mode = arg.fill_mode


data_gen_args = dict(
                    featurewise_center=True,
                    featurewise_std_normalization=True,
                    rotation_range=rotation_range,
                    width_shift_range=width_shift_range,
                    height_shift_range=height_shift_range,
                    #shear_range=0.05,
                    #zoom_range=zoom_range,
                    horizontal_flip=horizontal_flip,
                    fill_mode=fill_mode,
                    cval=0)


#draw the training process of every epoch
def show_train_history(train_history, train, loss, plt_save_name=plt_save_name):
    plt.plot(train_history.history['f1-score'])
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_f1-score'])
    plt.plot(train_history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.legend(['f1-score','loss', 'val_f1-score','val_loss'], loc='upper left')
    plt.title('Train hist')
    plt.savefig(plt_save_name)


##### training
trainGene = trainGenerator(batch_size, train_path, train_img_folder, train_label_folder, data_gen_args,flag_multi_class=False,save_to_dir = None)
valGene = trainGenerator(batch_size, val_path, val_img_folder, val_label_folder, data_gen_args,flag_multi_class=False,save_to_dir = None)

##### model
BACKBONE = 'resnet34'
#ENCODER_WEIGHTS = 'imagenet'
ENCODER_WEIGHTS = None
INPUT_SHAPE = (None, None, 1)
# preprocess_input = sm.get_preprocessing(BACKBONE)
sm.set_framework('tf.keras')
sm.framework()
base_model = sm.Unet(backbone_name=BACKBONE, encoder_weights=ENCODER_WEIGHTS)
inp = Input(shape=INPUT_SHAPE)
l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
out = base_model(l1)
model = Model(inp, out, name=base_model.name)
# loss = sm.losses.BinaryFocalLoss(gamma=2.0)
model.compile(
    'Adam',
    # loss='binary_crossentropy',
    loss=f1_loss,
    metrics=[sm.metrics.f1_score],
)
model_checkpoint = ModelCheckpoint(model_name, monitor='loss',verbose=1, save_best_only=True)

training = model.fit_generator(trainGene, steps_per_epoch=steps_per_epoch,
        epochs=epochs, validation_data=valGene, validation_steps=10, callbacks=[model_checkpoint])
show_train_history(training, 'accuracy', 'loss')

##### inference
model = load_model(model_name, compile=False)
testGene = testGenerator(test_img_path)
#testGene_for_eval = testGenerator_for_evaluation(test_img_path)
results = model.predict_generator(testGene, img_num, verbose=1)
#loss, acc = model.evaluate_generator(testGene_for_eval, steps=img_num, verbose=1)
#print("test loss:",loss,"  test accuracy:", acc)
#####


##### draw your inference results
if not os.path.exists(save_result_folder):
    os.makedirs(save_result_folder)

saveResult(save_result_folder, results, flag_multi_class=False)
#####


##### Record every command params of your training
if (os.path.isfile(csvfilename)!=True):
    csv_create(csvfilename, filenum, batch_size, steps_per_epoch, epochs, learning_rate, learning_decay_rate, rotation_range)
else:
    csv_append(csvfilename, filenum, batch_size, steps_per_epoch, epochs, learning_rate, learning_decay_rate, rotation_range)
#####


K.clear_session()




