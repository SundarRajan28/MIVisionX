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

#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_crop.h"
#include "parameter_crop.h"
#include "exception.h"

CropNode::CropNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs),
        _dest_width(_outputs[0]->info().max_shape()[0]),
        _dest_height(_outputs[0]->info().max_shape()[1])
{
    _crop_param = std::make_shared<RocalCropParam>(_batch_size);
}

void CropNode::create_node()
{
    if(_node)
        return;

    if(_dest_width == 0 || _dest_height == 0)
        THROW("Uninitialized destination dimension")

    _crop_param->create_array(_graph);

    int input_layout = (int)_inputs[0]->info().layout();
    int output_layout = (int)_outputs[0]->info().layout();
    int roi_type = (int)_inputs[0]->info().roi_type();
    vx_scalar in_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar out_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    // Create vx_tensor for the crop coordinates
    vx_size num_of_dims = 2;
    vx_size stride[num_of_dims];
    std::vector<size_t> crop_tensor_dims = {_batch_size, 4};
    stride[0] = sizeof(vx_uint32);
    stride[1] = stride[0] * crop_tensor_dims[0];
    vx_enum mem_type = VX_MEMORY_TYPE_HOST;
    if (_inputs[0]->info().mem_type() == RocalMemType::HIP)
        mem_type = VX_MEMORY_TYPE_HIP;
    allocate_host_or_pinned_mem(&_crop_coordinates, stride[1] * 4, _inputs[0]->info().mem_type());
    
    _crop_tensor = vxCreateTensorFromHandle(vxGetContext((vx_reference) _graph->get()), num_of_dims, crop_tensor_dims.data(), VX_TYPE_UINT32, 0, 
                                                                  stride, (void *)_crop_coordinates, mem_type);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_crop_tensor)) != VX_SUCCESS)
        THROW("Error: vxCreateTensorFromHandle(crop_tensor: failed " + TOSTR(status))
    
    _node = vxExtrppNode_Crop(_graph->get(), _inputs[0]->handle(), _crop_tensor, _outputs[0]->handle(),
                              in_layout_vx, out_layout_vx, roi_type_vx);
    
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the crop tensor (vxExtrppNode_Crop) failed: "+TOSTR(status))
}

void CropNode::update_node()
{
    _crop_param->set_image_dimensions((Rocal2DROI *)_inputs[0]->info().roi().get_ptr());
    _crop_param->update_array();
    std::vector<uint32_t> crop_h_dims, crop_w_dims;
    _crop_param->get_crop_dimensions(crop_w_dims, crop_h_dims);
    _outputs[0]->update_tensor_roi(crop_w_dims, crop_h_dims);
    
    // Obtain the crop coordinates and update the roi
    auto x1 = _crop_param->get_x1_arr_val();
    auto y1 = _crop_param->get_y1_arr_val();
    Rocal2DROI *src_roi = (Rocal2DROI *)_crop_coordinates;
    for(unsigned i = 0; i < _batch_size; i++) {
        src_roi[i].x1 = x1[i];
        src_roi[i].y1 = y1[i];
        src_roi[i].x2 = crop_w_dims[i];
        src_roi[i].y2 = crop_h_dims[i];
    }
}

void CropNode::init(unsigned int crop_h, unsigned int crop_w, float x_drift_, float y_drift_)
{
    _crop_param->crop_w = crop_w;
    _crop_param->crop_h = crop_h;
    _crop_param->x1     = x_drift_;
    _crop_param->y1     = y_drift_;
    FloatParam *x_drift  = ParameterFactory::instance()->create_single_value_float_param(x_drift_);
    FloatParam *y_drift  = ParameterFactory::instance()->create_single_value_float_param(y_drift_);
    _crop_param->set_x_drift_factor(core(x_drift));
    _crop_param->set_y_drift_factor(core(y_drift));
}

// This init is used only for centre crop
void CropNode::init(unsigned int crop_h, unsigned int crop_w)
{
    _crop_param->crop_w = crop_w;
    _crop_param->crop_h = crop_h;
    _crop_param->x1 = 0; 
    _crop_param->y1 = 0;
    _crop_param->set_fixed_crop(0.5, 0.5);    // for center_crop
}

void CropNode::init(FloatParam *crop_h_factor, FloatParam  *crop_w_factor, FloatParam *x_drift, FloatParam *y_drift)
{
    _crop_param->set_x_drift_factor(core(x_drift));
    _crop_param->set_y_drift_factor(core(y_drift));
    _crop_param->set_crop_height_factor(core(crop_h_factor));
    _crop_param->set_crop_width_factor(core(crop_w_factor));
    _crop_param->set_random();
}

CropNode::~CropNode() {
    if (_inputs[0]->info().mem_type() == RocalMemType::HIP) {
#if ENABLE_HIP
        hipError_t err = hipFree(_crop_coordinates);
        if(err != hipSuccess)
            std::cerr << "\n[ERR] hipFree failed  " << std::to_string(err) << "\n";
#endif
    } else {
        free(_crop_coordinates);
    }
    vxReleaseTensor(&_crop_tensor);
}
