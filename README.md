# Apache OpenWhisk runtimes for Cuda

### Local development
```
./gradlew core:cuda8Action:distDocker
```
This will produce the image `whisk/cuda8action`

Build and Push image
```
docker login
./gradlew core:cuda8Action:distDocker -PdockerImagePrefix=$prefix-user -PdockerRegistry=docker.io
```

