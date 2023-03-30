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

#include "commons.h"
#include "context.h"
#include "rocal_api.h"
#if ENABLE_OPENCL
#include "CL/cl.h"
#endif

RocalTensorList ROCAL_API_CALL
rocalGetOutputTensors(
                    RocalContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    try
    {
        return context->master_graph->get_output_tensors();
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
        return nullptr;
    }
    return nullptr;
}

void
ROCAL_API_CALL rocalSetOutputs(RocalContext p_context, unsigned int num_of_outputs, std::vector<RocalTensor> &output_images)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalSetOutputs")
    auto context = static_cast<Context *>(p_context);
    for (auto& it : output_images) {
        auto img = static_cast<RocalTensor>(it);
        context->master_graph->set_output(img);
    }
}
