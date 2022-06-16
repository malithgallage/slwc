from mode.config import *
from data import *
from skimage.io import imsave

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


trainGene = trainGenerator(batch_size, train_path, train_img_folder, train_label_folder, data_gen_args,flag_multi_class=False,save_to_dir = None)

i = 0
N = 10
os.makedirs('data/augmentations/image', exist_ok=True)
os.makedirs('data/augmentations/label', exist_ok=True)
for (img,mask) in trainGene:
  if i < N:
    for j in range(img.shape[0]):
      imsave(f'data/augmentations/image/{i}_{j}.png', img[j,:,:,:].reshape(256,256))
      imsave(f'data/augmentations/label/{i}_{j}.png', mask[j,:,:,:].reshape(256,256))
    i += 1
  else:
    break
