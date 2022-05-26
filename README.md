# object-detection-api

Make it easy to train and deploy Object Detection(SSD) and Image Segmentation(Mask R-CNN) Model Using [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

## Requirements

- [conda](https://docs.conda.io/en/latest/miniconda.html)
- [protoc](https://github.com/protocolbuffers/protobuf)

Use the following command to check if the installation is successful.

```shell
$ conda --version
conda 4.9.2
$ protoc --version
libprotoc 3.17.1
```

## Installation

### Conda

```zsh
$ conda create -n  od python=3.8.5 && conda activate od && make install
......
----------------------------------------------------------------------
Ran 24 tests in 21.869s

OK (skipped=1)
```

### Install directly (such as in [colab](https://colab.research.google.com/))

```zsh
$ make install
......
----------------------------------------------------------------------
Ran 24 tests in 21.869s

OK (skipped=1)
```

For details, please refer to [colab demo](./colab/Mask_R_CNN.ipynb).

> Note: If the installation fails, you can refer to the detailed steps in the [official document](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md).

> Note: The update of `cudaDNN` and `toolkit` may not be as fast as TensorFlow. Therefore, if your machine has a GPU, after the installation is complete, you need to downgrade TensorFlow to the version supported by `cudaDNN` and `toolkit` in order to support GPU training. Take `2.8.0` as an example:
>
> ```shell
> $ pip install --upgrade tf-models-official==2.8.0
> $ pip install --upgrade tensorflow==2.8.0
> ```

## Usage

### Train

#### Object Detection

[The easiest way to Train a Custom Object Detection Model Using TensorFlow Object Detection API](https://makeoptim.com/en/deep-learning/yiai-object-detection)

#### Image Segmentation

[The easiest way to Train a Custom Image Segmentation Model Using TensorFlow Object Detection API Mask R-CNN](https://makeoptim.com/en/deep-learning/yiai-image-segmentation)

### Deploy

[Deploy image segmentation (Mask R-CNN) model service with TensorFlow Serving & Flask](https://makeoptim.com/en/deep-learning/yiai-serving-flask-mask-rcnn)
