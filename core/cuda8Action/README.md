# Cuda Action

This image contains libraries and frameworks useful for running Cuda actions and is based on [8.0-runtime-ubuntu16.04](https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/8.0/runtime/Dockerfile)

Bellow are the versions for the included libraries:

| Cuda Version | Package | Notes |
| ------------- | ------- | ----- |
| 8.0.61  | Cuda-toolkit 8.0 | Based on Ubuntu 16.04


#### Cuda Action Sample

To view an example check the [sample/multiplying-arrays-with-cuda](./sample/README.md) and follow the instructions.

### 8.0.61 Details

#### Available Ubuntu packages

For a complete list execute:

```bash
$ docker run --rm --entrypoint apt docker5gmedia/cuda8action list --installed
```
