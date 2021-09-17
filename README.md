# object-detection-api

Make the use of [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) easier.

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

```zsh
$ make install
......
----------------------------------------------------------------------
Ran 24 tests in 21.869s

OK (skipped=1)
```

> Note: If the installation fails, you can refer to the detailed steps in the [official document](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md).

> Note: The update of `cudaDNN` and `toolkit` may not be as fast as TensorFlow. Therefore, if your machine has a GPU, after the installation is complete, you need to downgrade TensorFlow to the version supported by `cudaDNN` and `toolkit` in order to support GPU training. Take `2.4.1` as an example:
>
> ```shell
> $ pip install --upgrade tf-models-official==2.4.0
> $ pip install --upgrade tensorflow==2.4.1
> ```
