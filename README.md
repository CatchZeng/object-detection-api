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
