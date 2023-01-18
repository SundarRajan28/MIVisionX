/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#include <cmath>
#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <graph.h>
#include "parameter_random_crop.h"
#include "commons.h"

void RocalRandomCropParam::set_area_factor(Parameter<float>* crop_area_factor)
{
    if(!crop_area_factor)
        return ;
    ParameterFactory::instance()->destroy_param(area_factor);
    area_factor = crop_area_factor;
}

void RocalRandomCropParam::set_aspect_ratio(Parameter<float>* crop_aspect_ratio)
{
    if(!crop_aspect_ratio)
        return ;
    ParameterFactory::instance()->destroy_param(aspect_ratio);
    aspect_ratio = crop_aspect_ratio;
}

void RocalRandomCropParam::update_array()
{
    generate_random_seeds();
    fill_crop_dims();
    update_crop_array();
}

void RocalRandomCropParam::generate_random_seeds() {
    ParameterFactory::instance()->generate_seed(); // Renew and regenerate
    std::seed_seq seq{ParameterFactory::instance()->get_seed()};
    seq.generate(_seeds.begin(), _seeds.end());
}


void RocalRandomCropParam::fill_crop_dims()
{
    float crop_area_factor  = 1.0;
    float crop_aspect_ratio = 1.0;
    float in_ratio;
    unsigned short num_of_attempts = 5;
    float x_drift, y_drift;
    double target_area;
    auto is_valid_crop = [](uint h, uint w, uint height, uint width)
    {
        return (h < height && w < width);
    };
    std::uniform_real_distribution<float> _aspect_ratio_log_dis(std::log(3.0f / 4), std::log(4.0f / 3));
    std::uniform_real_distribution<float> _area_dis( 0.08, 1);
    float min_wh_ratio = 3.0f / 4;
    float max_wh_ratio = 4.0f / 3;
    float max_hw_ratio = 1 / min_wh_ratio;

    for(uint img_idx = 0; img_idx < batch_size; img_idx++)
    {
        _rand_gen.seed(_seeds[img_idx]);
        CropROI crop;
        int H = in_height[img_idx], W = in_width[img_idx];
        if (W <= 0 || H <= 0) {
            // Should not come here
            cropw_arr_val[img_idx] = crop.W;
            croph_arr_val[img_idx] = crop.H;
            x1_arr_val[img_idx] = crop.x;
            y1_arr_val[img_idx] = crop.y;
            x2_arr_val[img_idx] = x1_arr_val[img_idx] + cropw_arr_val[img_idx];
            y2_arr_val[img_idx] = y1_arr_val[img_idx] + croph_arr_val[img_idx];
        }        
        float min_area = W * H * _area_dis.a();
        int maxW = std::max<int>(1, H * max_wh_ratio);
        int maxH = std::max<int>(1, W * max_hw_ratio);
        // detect two impossible cases early
        if (H * maxW < min_area) {  // image too wide
        crop.set_shape(H, maxW);
        } else if (W * maxH < min_area) {  // image too tall
        crop.set_shape(maxH, W);
        } else { // it can still fail for very small images when size granularity matters
        int attempts_left = 100;
        for (; attempts_left > 0; attempts_left--) {
            float scale = _area_dis(_rand_gen);
            size_t original_area = H * W;
            float target_area = scale * original_area;
            float ratio = std::exp(_aspect_ratio_log_dis(_rand_gen));
            auto w = static_cast<int>(
                std::roundf(sqrtf(target_area * ratio)));
            auto h = static_cast<int>(
                std::roundf(sqrtf(target_area / ratio)));
            w = std::max(1, w);
            h = std::max(1, h);
            crop.set_shape(h, w);
            ratio = static_cast<float>(w) / h;
            if (w <= W && h <= H && ratio >= min_wh_ratio && ratio <= max_wh_ratio)
            break;
        }
        if (attempts_left <= 0) {
            float max_area = _area_dis.b() * W * H;
            float ratio = static_cast<float>(W) / H;
            if (ratio > max_wh_ratio) {
            crop.set_shape(H, maxW);
            } else if (ratio < min_wh_ratio) {
            crop.set_shape(maxH, W);
            } else {
            crop.set_shape(H, W);
            }
            float scale = std::min(1.0f, max_area / (crop.W * crop.H));
            crop.W = std::max<int>(1, crop.W * std::sqrt(scale));
            crop.H = std::max<int>(1, crop.H * std::sqrt(scale));
        }
        }
        crop.x = std::uniform_int_distribution<int>(0, W - crop.W)(_rand_gen);
        crop.y = std::uniform_int_distribution<int>(0, H - crop.H)(_rand_gen);

        cropw_arr_val[img_idx] = crop.W;
        croph_arr_val[img_idx] = crop.H;
        x1_arr_val[img_idx] = crop.x;
        y1_arr_val[img_idx] = crop.y;
        x2_arr_val[img_idx] = x1_arr_val[img_idx] + cropw_arr_val[img_idx];
        y2_arr_val[img_idx] = y1_arr_val[img_idx] + croph_arr_val[img_idx];
    }
}

Parameter<float> *RocalRandomCropParam::default_area_factor()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(AREA_FACTOR_RANGE[0],
                                                                         AREA_FACTOR_RANGE[1])->core;
}

Parameter<float> *RocalRandomCropParam::default_aspect_ratio()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(ASPECT_RATIO_RANGE[0],
                                                                         ASPECT_RATIO_RANGE[1])->core;
}
