## Introduction

This repository consists of training phase of CASA model to tune hyper parameters and to choose the right model architecture

## Improvements done on top of the previous code (Abdulmomen's)

- Add more arguments to choose:
  - choose the model type `[orig|abdulmomen's unet]`
  - the backbone / architecture `[vgg16|resnet34]`
  - weight initialization `[imagenet|random]`
  - metrics to be plotted/logged `[f1-score|accuracy]`
  - loss function `[binary_crossentropy|binary focal loss|f1-loss]`
  - gamma (this takes effect if loss function is binary focal loss)

### Available backbones

`['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152', 'seresnext50', 'seresnext101', 'senet154', 'resnext50', 'resnext101', 'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201', 'inceptionresnetv2', 'inceptionv3', 'mobilenet', 'mobilenetv2', 'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7']`

For more details, check [this](https://github.com/qubvel/segmentation_models)

## Experiments on puhti

- `mv run.def run`
- `chmod +x run`
- modify `run`
- `./run` to run the experiment
