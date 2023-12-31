"""
This module includes all the functions that are needed for
image transformations, data pre-processing and storing the results
in the directory.
"""

from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
from skimage import img_as_ubyte
import skimage.io as io
import skimage.transform as trans
import sys
from mode.config import *

np.set_printoptions(threshold=sys.maxsize, precision=5, suppress=True)

arg = command_arguments()
#########################configuration########################
sperm = [120, 0, 0]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([sperm, Unlabelled])
class_name = ['sperm', 'None']  # You must define by yourself

color = 'grayscale'

num_classes = 2

test_img_size = 256 * 256

img_size = (256, 256)


###############################################################


def adjustData(img, mask, flag_multi_class=False, num_class=None):
    """
        Normalizing the mask and thresholding the intensity values
        according to the number of classes. This was tested only when
        number of classes = 2
    """

    # TODO : Need to check this properly
    if (flag_multi_class):
        img = img / 255.
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
        mask[(mask != 0.) & (mask != 255.) & (mask != 128.)] = 0.
        new_mask = np.zeros(mask.shape + (num_class,))
        ########################################################################
        # You should define the value of your labelled gray imgs
        # For example,the imgs in /data/catndog/train/label/cat is labelled white
        # you got to define new_mask[mask == 255, 0] = 1
        # it equals to the one-hot array [1,0,0].
        ########################################################################
        new_mask[mask == 255., 0] = 1
        new_mask[mask == 128., 1] = 1
        new_mask[mask == 0., 2] = 1
        mask = new_mask

    else:
        if (np.max(img) > 1):
            img = img / 255.
        if (np.max(mask) > 1):
            mask = mask / 255.
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=True, num_class=num_classes, save_to_dir=None, target_size=img_size, seed=1):
    """
        Generate image and mask at the same time for training
        use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
        if you want to visualize the results of generator, set save_to_dir = "your path"
    """
    # Augmenting image data depending on the parameters
    image_datagen = ImageDataGenerator(**aug_dict)
    # Augmenting mask data depending on the parameters using the same seed.
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
        train_path + "image",
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path + "label",
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    print('classes:', image_generator.class_indices, mask_generator.class_indices)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield img, mask


def testGenerator(test_path, target_size=img_size, flag_multi_class=True, as_gray=True):
    """
    Preparing pre allocated images for testing
    """
    for impath in sorted(list(glob.glob('%s/*.png' % test_path)) + list(glob.glob('%s/*.jpg' % test_path))):
        img = io.imread(impath, as_gray=as_gray)
        if img.max() > 1:
            img = img / 255.
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if flag_multi_class else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


def labelVisualize(num_class, color_dict, img):
    """
    Visualising the images given the color values.
    """
    img_out = np.zeros(img[:, :, 0].shape + (3,))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            index_of_class = np.argmax(img[i, j])
            img_out[i, j] = color_dict[index_of_class]
    return img_out


def saveResult(save_path, npyfile, flag_multi_class=True, num_class=num_classes):
    """
    Saving the images in the given directory for visualization purpose
    """
    count = 1
    for i, item in enumerate(npyfile):
        if flag_multi_class:
            img = labelVisualize(num_class, COLOR_DICT, item)
            img = img.astype(np.uint8)
            io.imsave(os.path.join(save_path, "%d.png" % count), img)
        else:
            img = item[:, :, 0]
            print(np.max(img), np.min(img))
            img[img > 0.5] = 255
            img[img <= 0.5] = 0
            img = img.astype('uint8')
            print(np.max(img), np.min(img))
            # img = img_as_ubyte(img )
            io.imsave(os.path.join(save_path, "%d.png" % count), img)
        count += 1


def saveResult2(save_path, npyfile, test_path, flag_multi_class=True, num_class=num_classes, ):
    for impath, item in zip(list(sorted(glob.glob('%s/*.png' % test_path))), npyfile):
        if flag_multi_class:
            img = labelVisualize(num_class, COLOR_DICT, item)
            img = img.astype(np.uint8)
            imname = impath.split('/')[-1]
            io.imsave(os.path.join(save_path, imname), img)
            print(np.max(img), np.min(img))
        else:
            img = item[:, :, 0]
            print(np.max(img), np.min(img))
            img[img > 0.5] = 1
            img[img <= 0.5] = 0
            print(np.max(img), np.min(img))
            img = img * 255.
            img = img.astype(np.uint8)
            imname = impath.split('/')[-1]
            io.imsave(os.path.join(save_path, imname), img)
            print(imname)


def saveResult3(save_path, npyfile, test_path, flag_multi_class=True, num_class=num_classes):
    count = 1
    imgpaths = sorted(list(glob.glob('%s/*.png' % test_path)) + list(glob.glob('%s/*.jpg' % test_path)))
    for imgpath, item in zip(imgpaths, npyfile):
        imgname = os.path.basename(imgpath)[:-4]
        if flag_multi_class:
            img = labelVisualize(num_class, COLOR_DICT, item)
            img = img.astype(np.uint8)
            io.imsave(os.path.join(save_path, "%s.png" % imgname), img)
        else:
            img = item[:, :, 0]
            print(np.max(img), np.min(img))
            io.imsave(os.path.join(save_path, "%s_raw.png" % imgname), img)
            img[img > 0.5] = 255
            img[img <= 0.5] = 0
            img = img.astype('uint8')
            print(np.max(img), np.min(img))
            # img = img_as_ubyte(img )
            io.imsave(os.path.join(save_path, "%s.png" % imgname), img)
        count += 1
