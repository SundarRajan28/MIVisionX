/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include "internal_publishKernels.h"

struct SequenceRearrangeLocalData
{
    RPPCommonHandle handle;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    Rpp32u device_type;
    vx_uint32 new_sequence_length;
    vx_uint32 sequence_length;
    vx_uint32 *new_order;
    vx_enum in_tensor_type;
    vx_enum out_tensor_type;
    Rpp32u layout;
    RpptDescPtr src_desc_ptr;
    RpptDesc srcDesc;
    RpptDescPtr dst_desc_ptr;
    RpptDesc dstDesc;
#if ENABLE_OPENCL
    cl_mem cl_pSrc;
    cl_mem cl_pDst;
#endif
};

static vx_status VX_CALLBACK refreshSequenceRearrange(vx_node node, const vx_reference *parameters, vx_uint32 num, SequenceRearrangeLocalData *data)
{
    vx_status status = VX_SUCCESS;
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[2], 0, data->new_sequence_length, sizeof(vx_uint32), data->new_order, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_OPENCL
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->cl_pSrc, sizeof(data->cl_pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &data->cl_pDst, sizeof(data->cl_pDst)));
#elif ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->pDst, sizeof(data->pDst)));
#endif
    }
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
    {
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(data->pDst)));
    }
    return status;
}

static vx_status VX_CALLBACK validateSequenceRearrange(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // check scalar alpha and beta type
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #3 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #4 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #5 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #6 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[7], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #7 type=%d (must be size)\n", scalar_type);

    // Check for output parameters
    vx_tensor output;
    vx_parameter output_param;
    size_t num_tensor_dims;
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[RPP_MAX_TENSOR_DIMS];
    vx_enum tensor_type;
    output_param = vxGetParameterByIndex(node, 1);
    STATUS_ERROR_CHECK(vxQueryParameter(output_param, VX_PARAMETER_ATTRIBUTE_REF, &output, sizeof(vx_tensor)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    vxReleaseTensor(&output);
    vxReleaseParameter(&output_param);
    return status;
}

static vx_status VX_CALLBACK processSequenceRearrange(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    SequenceRearrangeLocalData *data = NULL;
    vx_status status = VX_SUCCESS;

    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_OPENCL
        cl_command_queue handle = data->handle.cmdq;
        refreshSequenceRearrange(node, parameters, num, data);
        for (int sequence_cnt = 0; sequence_cnt < data->src_desc_ptr->n; sequence_cnt++)
        {
            unsigned src_sequence_start_address = sequence_cnt * data->src_desc_ptr->strides.nStride * data->sequence_length;
            unsigned dst_sequence_start_address = sequence_cnt * data->dst_desc_ptr->strides.nStride * data->new_sequence_length;
            for (unsigned dst_index = 0; dst_index < (data->new_sequence_length); dst_index++)
            {
                unsigned src_index = data->new_order[dst_index];
                if (src_index > data->sequence_length)
                    ERRMSG(VX_ERROR_INVALID_VALUE, "invalid new order value=%d (must be between 0-%d)\n", src_index, data->sequence_length - 1);
                auto dst_offset = (unsigned char *)data->cl_pDst + dst_sequence_start_address + (dst_index * data->src_desc_ptr->strides.nStride);
                auto src_offset = (unsigned char *)data->cl_pSrc + src_sequence_start_address + (src_index * data->dst_desc_ptr->strides.nStride);
                if (clEnqueueCopyBuffer(handle, data->cl_pSrc, data->cl_pDst, src_offset, dst_offset, data->src_desc_ptr->strides.nStride, 0, NULL, NULL) != CL_SUCCESS)
                        return VX_FAILURE;
            }
        }
#elif ENABLE_HIP
        refreshSequenceRearrange(node, parameters, num, data);
        for (int sequence_cnt = 0; sequence_cnt < data->src_desc_ptr->n; sequence_cnt++)
        {
            unsigned src_sequence_start_address = sequence_cnt * data->src_desc_ptr->strides.nStride * data->sequence_length;
            unsigned dst_sequence_start_address = sequence_cnt * data->dst_desc_ptr->strides.nStride * data->new_sequence_length;
            for (unsigned dst_index = 0; dst_index < (data->new_sequence_length); dst_index++)
            {
                unsigned src_index = data->new_order[dst_index];
                if (src_index > data->sequence_length)
                    ERRMSG(VX_ERROR_INVALID_VALUE, "invalid new order value=%d (must be between 0-%d)\n", src_index, data->sequence_length - 1);
                auto dst_address = (unsigned char *)data->pDst + dst_sequence_start_address + (dst_index * data->src_desc_ptr->strides.nStride);
                auto src_address = (unsigned char *)data->pSrc + src_sequence_start_address + (src_index * data->dst_desc_ptr->strides.nStride);
                hipError_t status = hipMemcpyDtoD(dst_address, src_address, data->src_desc_ptr->strides.nStride);
                    if (status != hipSuccess)
                        return VX_FAILURE;  
            }
        }
#endif
    }
    else if (data->device_type == AGO_TARGET_AFFINITY_CPU)
    {
        refreshSequenceRearrange(node, parameters, num, data);
        for (int sequence_cnt = 0; sequence_cnt < data->src_desc_ptr->n; sequence_cnt++)
        {
            unsigned src_sequence_start_address = sequence_cnt * data->src_desc_ptr->strides.nStride * data->sequence_length;
            unsigned dst_sequence_start_address = sequence_cnt * data->dst_desc_ptr->strides.nStride * data->new_sequence_length;
            for (unsigned dst_index = 0; dst_index < (data->new_sequence_length); dst_index++)
            {
                unsigned src_index = data->new_order[dst_index];
                if (src_index > data->sequence_length)
                    ERRMSG(VX_ERROR_INVALID_VALUE, "invalid new order value=%d (must be between 0-%d)\n", src_index, data->sequence_length - 1);
                auto dst_address = (unsigned char *)data->pDst + dst_sequence_start_address + (dst_index * data->src_desc_ptr->strides.nStride);
                auto src_address = (unsigned char *)data->pSrc + src_sequence_start_address + (src_index * data->dst_desc_ptr->strides.nStride);
                memcpy(dst_address, src_address, data->src_desc_ptr->strides.nStride);
            }
        }
    }
    return status;
}

static vx_status VX_CALLBACK initializeSequenceRearrange(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    SequenceRearrangeLocalData *data = new SequenceRearrangeLocalData;
    memset(data, 0, sizeof(*data));
#if ENABLE_OPENCL
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE, &data->handle.cmdq, sizeof(data->handle.cmdq)));
#elif ENABLE_HIP
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &data->handle.hipstream, sizeof(data->handle.hipstream)));
#endif
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[7], &data->device_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[6], &data->layout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[3], &data->new_sequence_length, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[4], &data->sequence_length, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
 
    vx_size in_num_of_dims, out_num_of_dims;
    size_t in_tensor_dims[RPP_MAX_TENSOR_DIMS], out_tensor_dims[RPP_MAX_TENSOR_DIMS];
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &in_num_of_dims, sizeof(vx_size)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, in_tensor_dims, sizeof(vx_size) * in_num_of_dims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &data->in_tensor_type, sizeof(data->in_tensor_type)));

    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &out_num_of_dims, sizeof(vx_size)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, out_tensor_dims, sizeof(vx_size) * out_num_of_dims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &data->out_tensor_type, sizeof(data->out_tensor_type)));
    
    data->src_desc_ptr = &data->srcDesc;
    data->dst_desc_ptr = &data->dstDesc;
    if(data->layout == 2) // NFHWC
    {
        data->src_desc_ptr->n = in_tensor_dims[0];
        data->src_desc_ptr->h = in_tensor_dims[2];
        data->src_desc_ptr->w = in_tensor_dims[3];
        data->src_desc_ptr->c = in_tensor_dims[4];
        data->src_desc_ptr->strides.nStride = data->src_desc_ptr->c * data->src_desc_ptr->w * data->src_desc_ptr->h;

        data->dst_desc_ptr->n = out_tensor_dims[0];
        data->dst_desc_ptr->h = out_tensor_dims[2];
        data->dst_desc_ptr->w = out_tensor_dims[3];
        data->dst_desc_ptr->c = out_tensor_dims[4];
        data->dst_desc_ptr->strides.nStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w * data->dst_desc_ptr->h;
    }
    else if(data->layout == 3)// NFCHW
    {
        data->src_desc_ptr->n = in_tensor_dims[0];
        data->src_desc_ptr->h = in_tensor_dims[3];
        data->src_desc_ptr->w = in_tensor_dims[4];
        data->src_desc_ptr->c = in_tensor_dims[2];
        data->src_desc_ptr->strides.nStride = data->src_desc_ptr->c * data->src_desc_ptr->w * data->src_desc_ptr->h;
        data->dst_desc_ptr->n = out_tensor_dims[0];
        data->dst_desc_ptr->h = out_tensor_dims[3];
        data->dst_desc_ptr->w = out_tensor_dims[4];
        data->dst_desc_ptr->c = out_tensor_dims[2];
        data->dst_desc_ptr->strides.nStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w * data->dst_desc_ptr->h;
    }
    data->new_order = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->new_sequence_length);
    refreshSequenceRearrange(node, parameters, num, data);
    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeSequenceRearrange(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    SequenceRearrangeLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    free(data->new_order);
    delete (data);
    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
// TODO::currently the node is setting the same affinity as context. This needs to change when we have hubrid modes in the same graph
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
                                                  vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
                                                  vx_uint32 &supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
)
{
    vx_context context = vxGetContext((vx_reference)graph);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    else
        supported_target_affinity = AGO_TARGET_AFFINITY_CPU;

// hardcode the affinity to  CPU for OpenCL backend to avoid VerifyGraph failure since there is no codegen callback for amd_rpp nodes
#if ENABLE_OPENCL
    supported_target_affinity = AGO_TARGET_AFFINITY_CPU;
#endif
    return VX_SUCCESS;
}

vx_status SequenceRearrange_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.SequenceRearrange",
                                       VX_KERNEL_RPP_SEQUENCEREARRANGE,
                                       processSequenceRearrange,
                                       8,
                                       validateSequenceRearrange,
                                       initializeSequenceRearrange,
                                       uninitializeSequenceRearrange);
    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_OPENCL || ENABLE_HIP
    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));
#else
    vx_bool enableBufferAccess = vx_false_e;
#endif
    amd_kernel_query_target_support_f query_target_support_f = query_target_support;

    if (kernel)
    {
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));

        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS)
    {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }
    return status;
}
