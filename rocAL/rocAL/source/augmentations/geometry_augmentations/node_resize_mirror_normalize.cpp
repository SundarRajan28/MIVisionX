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
#include <cmath>
#include "node_resize_mirror_normalize.h"
#include "exception.h"

ResizeMirrorNormalizeNode::ResizeMirrorNormalizeNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs), _mirror(_mirror_range[0], _mirror_range[1])
{
}
    
void ResizeMirrorNormalizeNode::create_node()
{
    if(_node)
        return;
    std::vector<uint32_t> dst_roi_width(_batch_size,_outputs[0]->info().max_shape()[0]);
    std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().max_shape()[1]);
    
    std::vector<float> mean_vec, std_dev_vec;
    int mean_std_array_size = _batch_size * _inputs[0]->info().get_channels();
    if(!_std_dev[0])
        THROW("Standard deviation value cannot be 0");
    mean_vec.resize(mean_std_array_size, _mean[0]);
    std_dev_vec.resize(mean_std_array_size, _std_dev[0]);

    if(_inputs[0]->info().get_channels() == 3) {
        if(!(_std_dev[0] && _std_dev[1] && _std_dev[2]))
            THROW("Standard deviation value cannot be 0");
        for (uint i = 0, j = 0; i < _batch_size; i++, j += 3 ) {
            mean_vec[j ] = _mean[0];
            mean_vec[j + 1] = _mean[1];
            mean_vec[j + 2] = _mean[2];

            std_dev_vec[j ] = _std_dev[0];
            std_dev_vec[j + 1] = _std_dev[1];
            std_dev_vec[j + 2] = _std_dev[2];
        }
    }

    _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
   
    vx_status status = VX_SUCCESS;
    _mean_vx_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, mean_std_array_size);
    _std_dev_vx_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, mean_std_array_size);
    status |= vxAddArrayItems(_mean_vx_array, mean_std_array_size, mean_vec.data(), sizeof(vx_float32));
    status |= vxAddArrayItems(_std_dev_vx_array, mean_std_array_size, std_dev_vec.data(), sizeof(vx_float32));
    _mirror.create_array(_graph , VX_TYPE_UINT32, _batch_size);
    if(status != 0)
        THROW(" vxAddArrayItems failed in the resize_mirror_normalize node (vxExtrppNode_CropMirrorNormalize)  node: "+ TOSTR(status) + "  "+ TOSTR(status))

    vx_status width_status, height_status;

    width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
    if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the resize (vxExtrppNode_ResizebatchPD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status));

    int input_layout = (int)_inputs[0]->info().layout();
    int output_layout = (int)_outputs[0]->info().layout();
    int roi_type = (int)_inputs[0]->info().roi_type();
    vx_scalar in_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar out_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);
    vx_scalar interpolation_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_INT32,&_interpolation_type);
   _node = vxExtrppNode_ResizeMirrorNormalize(_graph->get(), _inputs[0]->handle(),
                                             _src_tensor_roi, _outputs[0]->handle(), _dst_roi_width, _dst_roi_height,
                                             interpolation_vx, _mean_vx_array, _std_dev_vx_array, _mirror.default_array(),
                                             in_layout_vx , out_layout_vx, roi_type_vx, _batch_size);
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the resize (vxExtrppNode_ResizeMirrorNormalize) node failed: "+ TOSTR(status))
}

void ResizeMirrorNormalizeNode::update_node()
{
    RocalROI* src_roi = _inputs[0]->info().get_roi();   // Check if it needs to be similar to resize
    for (unsigned i = 0; i < _batch_size; i++) {
        _src_width = src_roi[i].x2;
        _src_height = src_roi[i].y2;
        _dst_width = _out_width;
        _dst_height = _out_height;
        adjust_out_roi_size();
        _dst_width = std::min(_dst_width, (unsigned)_outputs[0]->info().max_shape()[0]);
        _dst_height = std::min(_dst_height, (unsigned)_outputs[0]->info().max_shape()[1]);
        _dst_roi_width_vec.push_back(_dst_width);
        _dst_roi_height_vec.push_back(_dst_height);
    }
    vx_status width_status, height_status;
    width_status = vxCopyArrayRange((vx_array)_dst_roi_width, 0, _batch_size, sizeof(vx_uint32), _dst_roi_width_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    height_status = vxCopyArrayRange((vx_array)_dst_roi_height, 0, _batch_size, sizeof(vx_uint32), _dst_roi_height_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(width_status != 0 || height_status != 0)
        WRN("ERROR: vxCopyArrayRange _dst_roi_width or _dst_roi_height failed " + TOSTR(width_status) + "  " + TOSTR(height_status));
    _outputs[0]->update_tensor_roi(_dst_roi_width_vec, _dst_roi_height_vec);
    _dst_roi_width_vec.clear();
    _dst_roi_height_vec.clear();
    _mirror.update_array();
}
void ResizeMirrorNormalizeNode::init(unsigned dest_width, unsigned dest_height, RocalResizeScalingMode scaling_mode, std::vector<unsigned> max_size,
                                     RocalResizeInterpolationType interpolation_type, std::vector<float>& mean, std::vector<float>& std_dev, IntParam *mirror)
{
    _interpolation_type = (int)interpolation_type;
    _scaling_mode = scaling_mode;
    _out_width = dest_width;
    _out_height = dest_height;
    if(max_size.size() > 0) {
        _max_width = max_size[0];
        _max_height = max_size[1];
    }
    _mean = mean;
    _std_dev = std_dev;
    _mirror.set_param(core(mirror));
}

void ResizeMirrorNormalizeNode::adjust_out_roi_size() {
    bool has_max_size = (_max_width | _max_height) > 0;

    if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_STRETCH) {
        if (_dst_width == 0) _dst_width = _src_width;
        if (_dst_height == 0) _dst_height = _src_height;

        if (has_max_size) {
            if (_max_width != 0) _dst_width = std::min(_dst_width, _max_width);
            if (_max_height != 0) _dst_height = std::min(_dst_height, _max_height);
        }
    } else if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_DEFAULT) {
        if (_dst_width == 0 && _dst_height != 0) {  // Only height is passed
            _dst_width = std::lround(_src_width * (static_cast<float>(_dst_height) / _src_height));
        } else if (_dst_height == 0 && _dst_width != 0) {  // Only width is passed
            _dst_height = std::lround(_src_height * (static_cast<float>(_dst_width) / _src_width));
        }

        if (has_max_size) {
            if (_max_width != 0) _dst_width = std::min(_dst_width, _max_width);
            if (_max_height != 0) _dst_height = std::min(_dst_height, _max_height);
        }
    } else {
        float scale = 1.0f;
        float scale_w = static_cast<float>(_dst_width) / _src_width;
        float scale_h = static_cast<float>(_dst_height) / _src_height;
        if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_SMALLER) {
            scale = std::max(scale_w, scale_h);
        } else if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_LARGER) {
            scale = (scale_w > 0 && scale_h > 0) ? std::min(scale_w, scale_h) : ((scale_w > 0) ? scale_w : scale_h);
        }

        if (has_max_size) {
            if (_max_width != 0) scale = std::min(scale, static_cast<float>(_max_width) / _src_width);
            if (_max_height != 0) scale = std::min(scale, static_cast<float>(_max_height) / _src_height);
        }

        if ((scale_w != scale) || (_dst_width == 0)) _dst_width = std::lround(_src_width * scale);
        if ((scale_h != scale) || (_dst_height == 0)) _dst_height = std::lround(_src_height * scale);
    }
}
