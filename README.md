# Apache OpenWhisk runtimes for Cuda

This action runtime enables developers to create Cuda based OpenWhisk actions. It comes with preinstalled Cuda libraries useful for running media-intensive and AI inferences. See more about [cuda8action](./core/cuda8Action).

### Local development
```
./gradlew core:cuda8Action:distDocker
```
This will produce the image `whisk/cuda8action`

Substitute `$prefix-user` with your docker hub username
```
docker login --username=$prefix-user
./gradlew core:cuda8Action:distDocker -PdockerImagePrefix=$prefix-user -PdockerRegistry=docker.io
```
This will build and push the image
