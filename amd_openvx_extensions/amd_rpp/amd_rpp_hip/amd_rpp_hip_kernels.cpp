/*
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "../../../amd_openvx/openvx/hipvx/hip_common_funcs.h"
#include "amd_rpp_hip_host_decls.h"

// multiplies entire input with a constant scalar value passed
__global__ void __attribute__((visibility("default")))
HipTensorMulScalar(const float *srcPtr,
                   float *dstPtr,
                   float scalarValue,
                   size_t maxTensorSize) 
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    if (id_x >= maxTensorSize)
        return;
    dstPtr[id_x] = srcPtr[id_x] * scalarValue;
}

int HipExecTensorMulScalar(hipStream_t stream,
                           const float *srcPtr,
                           float *dstPtr,
                           float scalarValue,
                           size_t maxTensorSize) 
{
    int localThreadsX = 256, localThreadsY = 1;
    int globalThreadsX = maxTensorSize, globalThreadsY = 1;
    hipLaunchKernelGGL(HipTensorMulScalar,
                       dim3(ceil((float)globalThreadsX / localThreadsX), ceil((float)globalThreadsY / localThreadsY)),
                       dim3(localThreadsX, localThreadsY),
                       0, 
                       stream, 
                       srcPtr, 
                       dstPtr, 
                       scalarValue, 
                       maxTensorSize);
    hipStreamSynchronize(stream);
    return VX_SUCCESS;
}

// Adds entire input with a constant scalar value passed
__global__ void __attribute__((visibility("default")))
HipTensorAddScalar(const float *srcPtr,
                   float *dstPtr,
                   float scalarValue,
                   size_t maxTensorSize) 
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    if (id_x >= maxTensorSize)
        return;
    dstPtr[id_x] = srcPtr[id_x] + scalarValue;
}

int HipExecTensorAddScalar(hipStream_t stream,
                           const float *srcPtr,
                           float *dstPtr,
                           float scalarValue,
                           size_t maxTensorSize) 
{
    int localThreadsX = 256, localThreadsY = 1;
    int globalThreadsX = maxTensorSize, globalThreadsY = 1;
    hipLaunchKernelGGL(HipTensorAddScalar,
                       dim3(ceil((float)globalThreadsX / localThreadsX), ceil((float)globalThreadsY / localThreadsY)),
                       dim3(localThreadsX, localThreadsY),
                       0, 
                       stream, 
                       srcPtr, 
                       dstPtr, 
                       scalarValue, 
                       maxTensorSize);
    hipStreamSynchronize(stream);
    return VX_SUCCESS;
}
