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



#include "augmentations_nodes.h"
#include "augmentations_meta_nodes.h"
#include "commons.h"
#include "context.h"
#include "rocal_api.h"
#include "image_source_evaluator.h"

// Calculated from the largest resize shorter dimension in imagenet validation dataset
#define MAX_ASPECT_RATIO 6.0f

RocalTensor  ROCAL_API_CALL
rocalSequenceRearrange(
            RocalContext p_context,
            RocalTensor input,
            unsigned int* new_order,
            unsigned int  new_sequence_length,
            unsigned int sequence_length,
            bool is_output)
{
    rocalTensor* output = nullptr;

    if ((p_context == nullptr) || (input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    try
    {

        if(sequence_length == 0)
            THROW("sequence_length passed should be bigger than 0")

        rocalTensorInfo output_info = input->info();
        std::vector<size_t> new_dims;
        new_dims = output_info.dims();
        new_dims[1] = new_sequence_length;
        output_info.set_dims(new_dims);

        output = context->master_graph->create_tensor(output_info, is_output);
        std::shared_ptr<SequenceRearrangeNode> sequence_rearrange_node =  context->master_graph->add_node<SequenceRearrangeNode>({input}, {output});
        sequence_rearrange_node->init(new_order, new_sequence_length, sequence_length, context->user_batch_size());
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalBrightness(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam p_alpha,
        RocalFloatParam p_beta,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    rocalTensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    auto beta = static_cast<FloatParam*>(p_beta);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        rocalTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<BrightnessNode>({input}, {output})->init(alpha, beta);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor  ROCAL_API_CALL
rocalCopyTensor(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    try
    {
        output = context->master_graph->create_tensor(input->info(), is_output);
        context->master_graph->add_node<CopyNode>({input}, {output});
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

// Commented for Now
// RocalTensor  ROCAL_API_CALL
// rocalNopTensor(
//         RocalContext p_context,
//         RocalTensor p_input,
//         bool is_output)
// {
//     if(!p_context || !p_input)
//         THROW("Null values passed as input")
//     rocalTensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     auto input = static_cast<rocalTensor*>(p_input);
//     try
//     {
//         output = context->master_graph->create_tensor(input->info(), is_output);
//         context->master_graph->add_tensor_node<NopNode>({input}, {output});
//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         ERR(e.what())
//     }
//     return output;
// }

RocalTensor ROCAL_API_CALL
rocalCrop(RocalContext p_context,
          RocalTensor p_input,
          bool is_output,
          RocalFloatParam p_crop_width,
          RocalFloatParam p_crop_height,
          RocalFloatParam p_crop_pox_x,
          RocalFloatParam p_crop_pos_y,
          RocalTensorLayout rocal_tensor_output_layout,
          RocalTensorOutputType rocal_tensor_output_datatype)
{
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    auto crop_h = static_cast<FloatParam*>(p_crop_height);
    auto crop_w = static_cast<FloatParam*>(p_crop_width);
    auto x_drift = static_cast<FloatParam*>(p_crop_pox_x);
    auto y_drift = static_cast<FloatParam*>(p_crop_pos_y);

    try
    {
        if(!input || !context)
            THROW("Null values passed as input")
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        rocalTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        // std::vector<size_t> out_dims = output_info.dims();
        // if(op_tensorLayout == RocalTensorlayout::NHWC) {
        //     out_dims[1] = crop_h;
        //     out_dims[2] = crop_w;
        // } else if(op_tensorLayout == RocalTensorlayout::NCHW) {
        //     out_dims[2] = crop_h;
        //     out_dims[3] = crop_w;
        // } else if(op_tensorLayout == RocalTensorlayout::NFHWC) {
        //     out_dims[2] = crop_h;
        //     out_dims[3] = crop_w;
        // } else if(op_tensorLayout == RocalTensorlayout::NFCHW) {
        //     out_dims[3] = crop_h;
        //     out_dims[4] = crop_w;
        // }
        // output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<CropNode> crop_node = context->master_graph->add_node<CropNode>({input}, {output});
        crop_node->init(crop_h, crop_w, x_drift, y_drift);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropMetaNode,CropNode>(crop_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }
    return output; // Changed to input----------------IMPORTANT
}

//resizemirrornormalize
RocalTensor
ROCAL_API_CALL rocalResizeMirrorNormalize(RocalContext p_context, 
                                          RocalTensor p_input,
                                          unsigned dest_width,
                                          unsigned dest_height,
                                          std::vector<float> &mean,
                                          std::vector<float> &std_dev,
                                          bool is_output,
                                          RocalResizeScalingMode scaling_mode,
                                          std::vector<unsigned> max_size,
                                          unsigned resize_shorter,
                                          unsigned resize_longer,
                                          RocalResizeInterpolationType interpolation_type,
                                          RocalIntParam p_mirror,
                                          RocalTensorLayout rocal_tensor_output_layout,
                                          RocalTensorOutputType rocal_tensor_output_datatype)
{
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    auto mirror = static_cast<IntParam *>(p_mirror);

    try
    {
        if((dest_width | dest_height | resize_longer | resize_shorter) == 0)
            THROW("Atleast one size 'dest_width' or 'dest_height' or 'resize_shorter' or 'resize_longer' must be specified")
        if((dest_width | dest_height) && (resize_longer | resize_shorter))
            THROW("Only one method of specifying size can be used \ndest_width and/or dest_height\nresize_shorter\nresize_longer")
        if(resize_longer && resize_shorter)
            THROW("'resize_longer' and 'resize_shorter' cannot be passed together. They are mutually exclusive.")

        unsigned out_width, out_height;
        RocalResizeScalingMode resize_scaling_mode;

        // Change the scaling mode if resize_shorter or resize_longer is specified
        if(resize_shorter) {
            resize_scaling_mode = RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_SMALLER;
            out_width = out_height = resize_shorter;
        } else if(resize_longer) {
            resize_scaling_mode = RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_LARGER;
            out_width = out_height = resize_longer;
        } else {
            resize_scaling_mode = scaling_mode;
            out_width = dest_width;
            out_height = dest_height;
        }

        std::vector<unsigned> maximum_size;
        if (max_size.size()) {
            if(max_size.size() == 1) {
                maximum_size = {max_size[0], max_size[0]};
            } else if(max_size.size() == 2) {
                maximum_size = {max_size[0], max_size[1]}; // {width, height}
            } else {
                THROW("The length of max_size vector exceeds the image dimension.")
            }
        }

        // Determine the max width and height to be set to the output info
        unsigned max_out_width, max_out_height;
        if (maximum_size.size() && maximum_size[0] != 0 && maximum_size[1] != 0) {
            // If max_size is passed by the user, the resized images cannot exceed the max size,
            max_out_width = maximum_size[0];
            max_out_height = maximum_size[1];
        } else {
            // compute the output info width and height wrt the scaling modes and roi passed
            if(resize_scaling_mode == ROCAL_SCALING_MODE_STRETCH) {
                max_out_width = out_width ? out_width : input->info().max_shape()[0];
                max_out_height = out_height ? out_height : input->info().max_shape()[1];
            } else if(resize_scaling_mode == ROCAL_SCALING_MODE_NOT_SMALLER) {
                max_out_width = (out_width ? out_width : out_height) * MAX_ASPECT_RATIO;
                max_out_height = (out_height ? out_height : out_width) * MAX_ASPECT_RATIO;
            } else {
                max_out_width = out_width ? out_width : out_height * MAX_ASPECT_RATIO;
                max_out_height = out_height ? out_height : out_width * MAX_ASPECT_RATIO;
            }
            if(maximum_size.size() == 2) {
                max_out_width = maximum_size[0] ? maximum_size[0] : max_out_width;
                max_out_height = maximum_size[1] ? maximum_size[1] : max_out_height;
            }
        }

        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        rocalTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensorLayout == RocalTensorlayout::NHWC)
        {
            out_dims[1] = max_out_height;
            out_dims[2] = max_out_width;
        }
        else if(op_tensorLayout == RocalTensorlayout::NCHW)
        {
            out_dims[2] = max_out_height;
            out_dims[3] = max_out_width;
        }
        else if(op_tensorLayout == RocalTensorlayout::NFHWC)
        {
            out_dims[2] = max_out_height;
            out_dims[3] = max_out_width;
        }
        else if(op_tensorLayout == RocalTensorlayout::NFCHW)
        {
            out_dims[3] = max_out_height;
            out_dims[4] = max_out_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<ResizeMirrorNormalizeNode> rmn_node = context->master_graph->add_node<ResizeMirrorNormalizeNode>({input}, {output});
        rmn_node->init(out_width, out_height, resize_scaling_mode, maximum_size, interpolation_type, mean, std_dev, mirror);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<ResizeMirrorNormalizeMetaNode,ResizeMirrorNormalizeNode>(rmn_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }

    return output; // Changed to input----------------IMPORTANT
}


RocalTensor  ROCAL_API_CALL
rocalCropFixed(
        RocalContext p_context,
        RocalTensor p_input,
        unsigned crop_width,
        unsigned crop_height,
        bool is_output,
        float crop_pos_x,
        float crop_pos_y,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype)
{
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);

    try
    {
        if(crop_width == 0 || crop_height == 0)
            THROW("Crop node needs tp receive non-zero destination dimensions")

        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        rocalTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        
        // For the crop node, user can create an image with a different width and height
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensorLayout == RocalTensorlayout::NHWC)
        {
            out_dims[1] = crop_height;
            out_dims[2] = crop_width;
        }
        else if(op_tensorLayout == RocalTensorlayout::NCHW)
        {
            out_dims[2] = crop_height;
            out_dims[3] = crop_width;
        }
        else if(op_tensorLayout == RocalTensorlayout::NFHWC)
        {
            out_dims[2] = crop_height;
            out_dims[3] = crop_width;
        }
        else if(op_tensorLayout == RocalTensorlayout::NFCHW)
        {
            out_dims[3] = crop_height;
            out_dims[4] = crop_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<CropNode> crop_node =  context->master_graph->add_node<CropNode>({input}, {output});
        crop_node->init(crop_height, crop_width, crop_pos_x, crop_pos_y);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropMetaNode,CropNode>(crop_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor  ROCAL_API_CALL
rocalCropCenterFixed(
        RocalContext p_context,
        RocalTensor p_input,
        unsigned crop_width,
        unsigned crop_height,
        bool is_output,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype)
{
    rocalTensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);

    try
    {
        if(crop_width == 0 || crop_height == 0)
            THROW("Crop node needs tp receive non-zero destination dimensions")
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        rocalTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        
        // For the crop node, user can create an image with a different width and height
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensorLayout == RocalTensorlayout::NHWC)
        {
            out_dims[1] = crop_height;
            out_dims[2] = crop_width;
        }
        else if(op_tensorLayout == RocalTensorlayout::NCHW)
        {
            out_dims[2] = crop_height;
            out_dims[3] = crop_width;
        }
        else if(op_tensorLayout == RocalTensorlayout::NFHWC)
        {
            out_dims[2] = crop_height;
            out_dims[3] = crop_width;
        }
        else if(op_tensorLayout == RocalTensorlayout::NFCHW)
        {
            out_dims[3] = crop_height;
            out_dims[4] = crop_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<CropNode> crop_node = context->master_graph->add_node<CropNode>({input}, {output});
        crop_node->init(crop_height, crop_width);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropMetaNode,CropNode>(crop_node);
    }

    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}



RocalTensor ROCAL_API_CALL 
rocalCropMirrorNormalize(RocalContext p_context, RocalTensor p_input, unsigned crop_height,
                         unsigned crop_width,float start_x, float start_y, std::vector<float> &mean,
                         std::vector<float> &std_dev, bool is_output, RocalIntParam p_mirror, 
                         RocalTensorLayout rocal_tensor_output_layout, RocalTensorOutputType rocal_tensor_output_datatype) {
    rocalTensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    auto mirror = static_cast<IntParam *>(p_mirror);
    try {
        if( crop_width == 0 || crop_height == 0)
            THROW("Null values passed as input")
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        rocalTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        
        // For the crop mirror normalize resize node, user can create an image with a different width and height
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensorLayout == RocalTensorlayout::NHWC) {
            out_dims[1] = crop_height;
            out_dims[2] = crop_width;
        } else if(op_tensorLayout == RocalTensorlayout::NCHW) {
            out_dims[2] = crop_height;
            out_dims[3] = crop_width;
        } else if(op_tensorLayout == RocalTensorlayout::NFHWC) {
            out_dims[2] = crop_height;
            out_dims[3] = crop_width;
        } else if(op_tensorLayout == RocalTensorlayout::NFCHW) {
            out_dims[3] = crop_height;
            out_dims[4] = crop_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<CropMirrorNormalizeNode> cmn_node = context->master_graph->add_node<CropMirrorNormalizeNode>({input}, {output});
        cmn_node->init(crop_height, crop_width, start_x, start_y, mean, std_dev , mirror);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropMirrorNormalizeMetaNode,CropMirrorNormalizeNode>(cmn_node);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


RocalTensor ROCAL_API_CALL
rocalResize(RocalContext p_context,
            RocalTensor p_input,
            unsigned dest_width,
            unsigned dest_height,
            bool is_output,
            RocalResizeScalingMode scaling_mode,
            std::vector<unsigned> max_size,
            unsigned resize_shorter,
            unsigned resize_longer,
            RocalResizeInterpolationType interpolation_type,
            RocalTensorLayout rocal_tensor_output_layout,
            RocalTensorOutputType rocal_tensor_output_datatype)
{
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);

    try
    {
        if((dest_width | dest_height | resize_longer | resize_shorter) == 0)
            THROW("Atleast one size 'dest_width' or 'dest_height' or 'resize_shorter' or 'resize_longer' must be specified")
        if((dest_width | dest_height) && (resize_longer | resize_shorter))
            THROW("Only one method of specifying size can be used \ndest_width and/or dest_height\nresize_shorter\nresize_longer")
        if(resize_longer && resize_shorter)
            THROW("'resize_longer' and 'resize_shorter' cannot be passed together. They are mutually exclusive.")

        unsigned out_width, out_height;
        RocalResizeScalingMode resize_scaling_mode;

        // Change the scaling mode if resize_shorter or resize_longer is specified
        if(resize_shorter) {
            resize_scaling_mode = RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_SMALLER;
            out_width = out_height = resize_shorter;
        } else if(resize_longer) {
            resize_scaling_mode = RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_LARGER;
            out_width = out_height = resize_longer;
        } else {
            resize_scaling_mode = scaling_mode;
            out_width = dest_width;
            out_height = dest_height;
        }

        std::vector<unsigned> maximum_size;
        if (max_size.size()) {
            if(max_size.size() == 1) {
                maximum_size = {max_size[0], max_size[0]};
            } else if(max_size.size() == 2) {
                maximum_size = {max_size[0], max_size[1]}; // {width, height}
            } else {
                THROW("The length of max_size vector exceeds the image dimension.")
            }
        }

        // Determine the max width and height to be set to the output info
        unsigned max_out_width, max_out_height;
        if (maximum_size.size() && maximum_size[0] != 0 && maximum_size[1] != 0) {
            // If max_size is passed by the user, the resized images cannot exceed the max size,
            max_out_width = maximum_size[0];
            max_out_height = maximum_size[1];
        } else {
            // compute the output info width and height wrt the scaling modes and roi passed
            if(resize_scaling_mode == ROCAL_SCALING_MODE_STRETCH) {
                max_out_width = out_width ? out_width : input->info().max_shape()[0];
                max_out_height = out_height ? out_height : input->info().max_shape()[1];
            } else if(resize_scaling_mode == ROCAL_SCALING_MODE_NOT_SMALLER) {
                max_out_width = (out_width ? out_width : out_height) * MAX_ASPECT_RATIO;
                max_out_height = (out_height ? out_height : out_width) * MAX_ASPECT_RATIO;
            } else {
                max_out_width = out_width ? out_width : out_height * MAX_ASPECT_RATIO;
                max_out_height = out_height ? out_height : out_width * MAX_ASPECT_RATIO;
            }
            if(maximum_size.size() == 2) {
                max_out_width = maximum_size[0] ? maximum_size[0] : max_out_width;
                max_out_height = maximum_size[1] ? maximum_size[1] : max_out_height;
            }
        }

        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        rocalTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensorLayout == RocalTensorlayout::NHWC)
        {
            out_dims[1] = max_out_height;
            out_dims[2] = max_out_width;
        }
        else if(op_tensorLayout == RocalTensorlayout::NCHW)
        {
            out_dims[2] = max_out_height;
            out_dims[3] = max_out_width;
        }
        else if(op_tensorLayout == RocalTensorlayout::NFHWC)
        {
            out_dims[2] = max_out_height;
            out_dims[3] = max_out_width;
        }
        else if(op_tensorLayout == RocalTensorlayout::NFCHW)
        {
            out_dims[3] = max_out_height;
            out_dims[4] = max_out_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<ResizeNode> resize_node =  context->master_graph->add_node<ResizeNode>({input}, {output});
        resize_node->init(out_width, out_height, resize_scaling_mode, maximum_size, interpolation_type);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<ResizeMetaNode,ResizeNode>(resize_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }
    return output; // Changed to input----------------IMPORTANT
}


RocalTensor ROCAL_API_CALL
rocalColorTwist(RocalContext p_context,
                RocalTensor p_input,
                bool is_output,
                RocalFloatParam p_alpha,
                RocalFloatParam p_beta,
                RocalFloatParam p_hue,
                RocalFloatParam p_sat,
                RocalTensorLayout rocal_tensor_output_layout,
                RocalTensorOutputType rocal_tensor_output_datatype)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    auto beta = static_cast<FloatParam*>(p_beta);
    auto hue = static_cast<FloatParam*>(p_hue);
    auto sat = static_cast<FloatParam*>(p_sat);

    try
    {
        int layout=0;
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        rocalTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ColorTwistNode>({input}, {output})->init(alpha, beta, hue ,sat, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}
