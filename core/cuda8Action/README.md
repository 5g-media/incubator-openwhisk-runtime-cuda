# Cuda action example

This example will guide you to create an action that adds elements of two arrays with a million elements each

Example code is taken from a [easier-introduction-to-cuda](https://devblogs.nvidia.com/even-easier-introduction-cuda/) blog.

## Prepare the development environment

### Run the development container

**Note:** You must run it from GPU node that contains the required Nvidia/Cuda drivers and nvidia-docker runtime (TODO: link)

This command runs the container passing it the local `~/.wskprops` to use Apache OpenWhisk credentials within.

```bash
docker run -it -e OPENWHISK_AUTH=`cat ~/.wskprops | grep ^AUTH= | awk -F= '{print $2}'` -e OPENWHISK_APIHOST=`cat ~/.wskprops | grep ^APIHOST= | awk -F= '{print $2}'` --rm nvidia/cuda:8.0-devel-ubuntu16.04 /bin/bash
```

### Install required packages

```bash
apt-get update && apt-get install -y vim curl zip
```

### Install Apache OpenWhisk CLI

```bash
curl -L https://github.com/apache/incubator-openwhisk-cli/releases/download/latest/OpenWhisk_CLI-latest-linux-amd64.tgz -o /tmp/wsk.tgz
tar xvfz /tmp/wsk.tgz -C /tmp/
mv /tmp/wsk /usr/local/bin
```

### Configure CLI

Refer to [configure-the-wsk-cli](https://github.com/apache/incubator-openwhisk-deploy-kube#configure-the-wsk-cli). This guide
refers to `mycluster.yaml` of your OpenWhisk deployment.

### Create the example

Copy the below block into `add.cu` 

**Note:** The application was slightly modified to return stringified JSON to be interpreted as
an OpenWhisk action.

```
#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y;

  // Allocate Unified Memory . accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "{\"message\": \"Max error: " << maxError << "\"}";

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
```

### Compile and test the code

```bash
nvcc add.cu -o add_cuda
./add_cuda
```

Program should return `{"message": "Max error: 0"}`

### Create initialization data via a (zip) file, similar to other actions kinds 

```bash
mv add_cuda exec
zip myAction.zip ./exec
```

### Create and invoke the action

```bash
wsk action create cuda_Test myAction.zip --kind cuda:8@selector
wsk -b action invoke cuda_Test
```
 
