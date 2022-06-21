from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    MaxPooling2D,
    UpSampling2D,
    Dropout,
    concatenate
)
from tensorflow.keras.optimizers import Adam
from mode.config import command_arguments
import segmentation_models as sm
from custom_losses import f1_loss, f1_loss2

arg = command_arguments()
learning_rate = arg.learning_rate
learning_decay_rate = arg.learning_decay_rate
loss = arg.loss
metric = arg.metric
if loss == "binary_focal_loss":
    gamma = arg.gamma
    loss = sm.losses.BinaryFocalLoss(gamma=gamma)
elif loss == "f1_loss":
    loss = f1_loss
elif loss == "f1_loss2":
    loss = f1_loss2

if metric == "f1-score":
    metric = sm.metrics.f1_score

backbone = arg.backbone
encoder_weights = arg.encoder_weights
if encoder_weights == "None":
    encoder_weights = None

img_size = (256, 256, 1)
dr_rate = 0.6
leakyrelu_alpha = 0.3


def unet(pretrained_weights=None, input_size=img_size):
    inputs = Input(input_size)
    conv1 = Conv2D(
        64, 3, activation=None, padding="same", kernel_initializer="he_normal"
    )(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=leakyrelu_alpha)(conv1)
    conv1 = Conv2D(
        64, 3, activation=None, padding="same", kernel_initializer="he_normal"
    )(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=leakyrelu_alpha)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(
        128, 3, activation=None, padding="same", kernel_initializer="he_normal"
    )(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=leakyrelu_alpha)(conv2)
    conv2 = Conv2D(
        128, 3, activation=None, padding="same", kernel_initializer="he_normal"
    )(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=leakyrelu_alpha)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(
        256, 3, activation=None, padding="same", kernel_initializer="he_normal"
    )(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=leakyrelu_alpha)(conv3)
    conv3 = Conv2D(
        256, 3, activation=None, padding="same", kernel_initializer="he_normal"
    )(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=leakyrelu_alpha)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(
        512, 3, activation=None, padding="same", kernel_initializer="he_normal"
    )(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=leakyrelu_alpha)(conv4)
    conv4 = Conv2D(
        512, 3, activation=None, padding="same", kernel_initializer="he_normal"
    )(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=leakyrelu_alpha)(conv4)
    drop4 = Dropout(dr_rate)(conv4)  ###
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(
        1024, 3, activation=None, padding="same", kernel_initializer="he_normal"
    )(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=leakyrelu_alpha)(conv5)
    conv5 = Conv2D(
        1024, 3, activation=None, padding="same", kernel_initializer="he_normal"
    )(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = LeakyReLU(alpha=leakyrelu_alpha)(conv5)

    up6 = Conv2D(
        512, 2, activation=None, padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(drop5))
    up6 = LeakyReLU(alpha=leakyrelu_alpha)(up6)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(
        512, 3, activation=None, padding="same", kernel_initializer="he_normal"
    )(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU(alpha=leakyrelu_alpha)(conv6)
    conv6 = Conv2D(
        512, 3, activation=None, padding="same", kernel_initializer="he_normal"
    )(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU(alpha=leakyrelu_alpha)(conv6)

    up7 = Conv2D(
        256, 2, activation=None, padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(conv6))
    up7 = BatchNormalization()(up7)
    up7 = LeakyReLU(alpha=leakyrelu_alpha)(up7)
    up7 = Dropout(dr_rate)(up7)  ###
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(
        256, 3, activation=None, padding="same", kernel_initializer="he_normal"
    )(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(alpha=leakyrelu_alpha)(conv7)
    conv7 = Conv2D(
        256, 3, activation=None, padding="same", kernel_initializer="he_normal"
    )(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(alpha=leakyrelu_alpha)(conv7)

    up8 = Conv2D(
        128, 2, activation=None, padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(conv7))
    up8 = BatchNormalization()(up8)
    up8 = LeakyReLU(alpha=0.3)(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(
        128, 3, activation=None, padding="same", kernel_initializer="he_normal"
    )(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU(alpha=0.3)(conv8)
    conv8 = Conv2D(
        128, 3, activation=None, padding="same", kernel_initializer="he_normal"
    )(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU(alpha=0.3)(conv8)

    up9 = Conv2D(
        64, 2, activation=None, padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(conv8))
    up9 = BatchNormalization()(up9)
    up9 = LeakyReLU(alpha=leakyrelu_alpha)(up9)
    up9 = Dropout(dr_rate)(up9)  ###
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(
        64, 3, activation=None, padding="same", kernel_initializer="he_normal"
    )(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = LeakyReLU(alpha=leakyrelu_alpha)(conv9)
    conv9 = Conv2D(
        64, 3, activation=None, padding="same", kernel_initializer="he_normal"
    )(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = LeakyReLU(alpha=leakyrelu_alpha)(conv9)
    conv9 = Conv2D(
        2, 3, activation=None, padding="same", kernel_initializer="he_normal"
    )(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = LeakyReLU(alpha=leakyrelu_alpha)(conv9)
    # Changed this from 3, 1 to 1, 1, which denotes the number of classes
    conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    loss = sm.losses.BinaryFocalLoss(gamma=2.0)
    model.compile(
        optimizer=Adam(lr=learning_rate, decay=learning_decay_rate),
        loss=loss,
        metrics=[metric],
    )

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def segmod():
    INPUT_SHAPE = (None, None, 1)
    sm.set_framework("tf.keras")
    sm.framework()
    base_model = sm.Unet(backbone_name=backbone, encoder_weights=encoder_weights)
    inp = Input(shape=INPUT_SHAPE)
    # map N channels data to 3 channels
    l1 = Conv2D(3, (1, 1))(inp)
    out = base_model(l1)
    model = Model(inp, out, name=base_model.name)
    model.compile(
        optimizer=Adam(lr=learning_rate, decay=learning_decay_rate),
        loss=loss,
        metrics=[metric],
    )
    return model
