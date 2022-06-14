## Introduction

This repository consists of training phase of CASA model to tune hyper parameters and to choose the right model architecture

## Improvements done on top of the previous code (Abdulmomen's)

- Add more arguments to choose:
  - the backbone / architecture `[vgg16|resnet34|abdulmomen's unet]`
  - weight initialization `[imagenet|random]`
  - metrics to be plotted/logged `[f1-score|accuracy]`
  - loss function `[binary_crossentropy|binary focal loss|f1-loss]`
  - gamma (this takes effect if loss function is binary focal loss)

## Experiments on puhti

- `mv run.def run`
- `chmod +x run`
- modify `run`
- `./run` to run the experiment
