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
#include "node_color_twist.h"
#include "exception.h"

ColorTwistNode::ColorTwistNode(const std::vector<rocalTensor *> &inputs,const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs),
        _alpha(ALPHA_RANGE[0], ALPHA_RANGE[1]),
        _beta (BETA_RANGE[0], BETA_RANGE[1]),
        _hue(HUE_RANGE[0], HUE_RANGE[1]),
        _sat(SAT_RANGE[0], SAT_RANGE[1])
{
}

void ColorTwistNode::create_node()
{
    if(_node)
        return;

    _alpha.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _beta.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _hue.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _sat.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);

    // if(_inputs[0]->info().layout() == RocalTensorlayout::NCHW)
    //     _layout = 1;
    // else if(_inputs[0]->info().layout() == RocalTensorlayout::NFHWC)
    //     _layout = 2;
    // else if(_inputs[0]->info().layout() == RocalTensorlayout::NFCHW)
    //     _layout = 3;
    if(_inputs[0]->info().roi_type() == RocalROIType::XYWH)
        _roi_type = 1;
    vx_scalar layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_layout);
    vx_scalar roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_roi_type);

    _node = vxExtrppNode_ColorTwist(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), _alpha.default_array(), _beta.default_array(), _hue.default_array(), _sat.default_array(), layout, roi_type, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the colortwist_batch (vxExtrppNode_ColotTwsit) node failed: "+ TOSTR(status))
}

void ColorTwistNode::init( float alpha, float beta, float hue , float sat, int layout)
{
    _alpha.set_param(alpha);
    _beta.set_param(beta);
    _hue.set_param(hue);
    _sat.set_param(sat);
    _layout = _roi_type = 0;
}

void ColorTwistNode::init( FloatParam* alpha, FloatParam* beta, FloatParam* hue, FloatParam* sat, int layout)
{
    _alpha.set_param(core(alpha));
    _beta.set_param(core(beta));
    _hue.set_param(core(hue));
    _sat.set_param(core(sat));
    _layout = _roi_type = 0;
}


void ColorTwistNode::update_node()
{
    _alpha.update_array();
    _beta.update_array();
    _hue.update_array();
    _sat.update_array();
}
