/*
MIT License

Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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


#include <iostream>
#include <cstring>
#include <chrono>
#include <cstdio>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include <opencv2/opencv.hpp>
using namespace cv;

#if USE_OPENCV_4
#define  CV_FONT_HERSHEY_DUPLEX    FONT_HERSHEY_DUPLEX
#define  CV_WINDOW_AUTOSIZE        WINDOW_AUTOSIZE
#define  CV_RGB2BGR                cv::COLOR_BGR2RGB
#else
#include <opencv/highgui.h>
#endif

#include "rocal_api.h"

#define DISPLAY
using namespace std::chrono;


int main(int argc, const char ** argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if(argc < MIN_ARG_COUNT) {
        printf( "Usage: image_augmentation <image_dataset_folder/video_file> <processing_device=1/cpu=0>  \
              decode_width decode_height video_mode gray_scale/rgb display_on_off decode_shard_count  <shuffle:0/1> <jpeg_dec_mode<0(tjpeg)/1(opencv)/2(hwdec)>\n" );
        return -1;
    }
    int argIdx = 0;
    const char * folderPath1 = argv[++argIdx];
    const char * folderPath2 = argv[++argIdx];
    int rgb = 1;// process color images
    int decode_width = 1000;
    int decode_height = 1000;
    bool processing_device = 0;
    size_t shard_count = 1;
    int shuffle = 1;
    int dec_mode = 0;
    bool print_hip_values = 1;

    int inputBatchSize = 64;
    // float *image_output, *image_output1;
    // image_output = (float *)malloc(inputBatchSize * 224 * 224 * 3 * sizeof(float));
    // image_output1 = (float *)malloc(inputBatchSize * 224 * 224 * 3 * sizeof(float));

    // Allocate device buffers
    float *d_image_output, *d_image_output1;
    int *d_labels_output, *d_val_labels_output;
    // if(processing_device)
    // {
    hipMalloc(&d_image_output, inputBatchSize * 224 * 224 * 3 * sizeof(float));
    // hipMalloc(&d_image_output1, inputBatchSize * 224 * 224 * 3 * sizeof(float));
    hipMalloc(&d_labels_output, inputBatchSize * sizeof(int));
    // hipMalloc(&d_val_labels_output, inputBatchSize * sizeof(int));
    std::cerr<<"allocated device buffers"<<std::endl;
    // }

    std::cout << ">>> Running on " << (processing_device?"GPU":"CPU") << std::endl;

    RocalImageColor color_format = RocalImageColor::ROCAL_COLOR_RGB24;
    RocalContext handle;
    handle = rocalCreate(inputBatchSize, processing_device?RocalProcessMode::ROCAL_PROCESS_GPU:RocalProcessMode::ROCAL_PROCESS_CPU, 0,1,6);
    rocalSetSeed(10);

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not create the rocAL contex\n";
        return -1;
    }

    RocalDecoderType dec_type = (RocalDecoderType) dec_mode;
    RocalTensor input1, input2;
    RocalMetaData metadata_output;
    RocalTensor image0, image1;

    RocalTensorLayout tensorLayout = RocalTensorLayout::ROCAL_NHWC;
    RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_UINT8;

    /*>>>>>>>>>>>>>>>> Creating rocAL parameters  <<<<<<<<<<<<<<<<*/

    // Creating a custom random object to set a limited number of values to randomize the rotation angle
    std::vector<int> values = {0, 1};
    std::vector<double> frequencies = {0.5, 0.5};
    RocalIntParam mirror = rocalCreateIntRand(values.data(), frequencies.data(), values.size());


    // /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/
    std::vector<float> random_area = {0.08, 1.0};
    std::vector<float> random_aspect_ratio = {0.8, 1.25};


    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    if (dec_type == RocalDecoderType::ROCAL_DECODER_OPENCV) std::cout << "Using OpenCV decoder for Jpeg Source\n";
    metadata_output = rocalCreateLabelReader(handle, folderPath1);
    input1 = rocalFusedJpegCropSingleShard(handle, folderPath1, color_format, 0, shard_count, false, random_area, random_aspect_ratio, 100, true, false,  ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_width, decode_height);


    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "JPEG source could not initialize : "<<rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    int resize_w = 224, resize_h = 224;

    // image0 = rocalResize(handle, input1, resize_w, resize_h, false);
    // image0 = rocalResize(handle, input1, resize_w, resize_h, false, ROCAL_SCALING_MODE_DEFAULT, {}, 0, 0, ROCAL_TRIANGULAR_INTERPOLATION);

    std::vector<unsigned> max_size;
    std::vector<float> mean = {0.485 * 255,0.456 * 255,0.406 * 255};
    std::vector<float> std_dev = {0.229 * 255,0.224 * 255,0.225 * 255};
    // image1 = rocalCropMirrorNormalize(handle, image0, 0, 224, 224, 1, 1, 1, mean, std_dev, true, output_array);

    image0 = rocalResize(handle, input1, resize_w, resize_h, false, ROCAL_SCALING_MODE_DEFAULT, max_size, 0, 0, ROCAL_TRIANGULAR_INTERPOLATION, RocalTensorLayout::ROCAL_NHWC, tensorOutputType);
    image1 = rocalCropMirrorNormalize(handle, image0, 224, 224, 0.5, 0.5, mean, std_dev, true, mirror, tensorLayout, RocalTensorOutputType::ROCAL_FP32);

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Error while adding the augmentation nodes " << std::endl;
        auto err_msg = rocalGetErrorMessage(handle);
        std::cout << err_msg << std::endl;
    }
    // Calling the API to verify and build the augmentation graph
    if(rocalVerify(handle) != ROCAL_OK)
    {
        std::cout << "Could not verify the augmentation graph" << std::endl;
        return -1;
    }

    std::cout << "Remaining images " << rocalGetRemainingImages(handle) << std::endl;
    // std::cout << "Augmented copies count " << rocalGetAugmentationBranchCount(handle) << std::endl;

    // auto handle1 = rocalCreate(inputBatchSize, processing_device?RocalProcessMode::ROCAL_PROCESS_GPU:RocalProcessMode::ROCAL_PROCESS_CPU, 0,1,2);
    // rocalSetSeed(10);

    // if(rocalGetStatus(handle1) != ROCAL_OK)
    // {
    //     std::cout << "Could not create the rocAL contex\n";
    //     return -1;
    // }

    // rocalCreateLabelReader(handle1, folderPath2);
    // input2 = rocalJpegFileSourceSingleShard(handle1, folderPath2, color_format, 0, 1, false, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, 1000, 1000);

    // RocalImage image2, image3, image4;
    // int resize_shorter = 256;

    // image2 = rocalResize(handle1, input2, 0, 0, false, ROCAL_SCALING_MODE_NOT_SMALLER, {}, resize_shorter, 0, ROCAL_TRIANGULAR_INTERPOLATION);
    // // image2 = rocalResize(handle1, input2, 224, 224, false, ROCAL_SCALING_MODE_DEFAULT, interpolation_type = ROCAL_TRIANGULAR_INTERPOLATION);
    // image3 = rocalCropCenterFixed(handle1, image2, 224, 224, 1, false);
    // image4 = rocalCropMirrorNormalize(handle1, image3, 0, 224, 224, 1, 1, 1, mean, std_dev, true);

    // if(rocalGetStatus(handle1) != ROCAL_OK)
    // {
    //     std::cout << "Error while adding the augmentation nodes " << std::endl;
    //     auto err_msg = rocalGetErrorMessage(handle1);
    //     std::cout << err_msg << std::endl;
    // }
    // // Calling the API to verify and build the augmentation graph
    // if(rocalVerify(handle1) != ROCAL_OK)
    // {
    //     std::cout << "Could not verify the augmentation graph" << std::endl;
    //     return -1;
    // }

    // std::cout << "Remaining images " << rocalGetRemainingImages(handle1) << std::endl;
    // std::cout << "Augmented copies count " << rocalGetAugmentationBranchCount(handle1) << std::endl;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    RocalTensorList output_tensor_list, labels;
    for (int epoch = 0; epoch < 1; epoch++)
    {
        // int counter = 0;
        std::cerr << "Epoch " << epoch+1 << std::endl;
        double acc_time = 0.0;
        double memcpy_acc_time = 0.0;
        double metadata_acc_time = 0.0;
        high_resolution_clock::time_point c1, c2;
        while (!rocalIsEmpty(handle))
        {
            if(rocalRun(handle) != 0)
                break;
            // if(processing_device)
            // {
            //     // rocalCopyToOutputTensor(handle, (void*)d_image_output, RocalTensorLayout::ROCAL_NCHW, RocalTensorOutputType::ROCAL_FP32, 0.485 * 255,0.456 * 255,0.406 * 255, 0.229 * 255,0.224 * 255,0.225 * 255, false);
            //     // rocalGetImageLabels(handle,(void*)d_labels_output);
            // }
            // else
            // {
                // int labels_output[inputBatchSize];
                // rocalCopyToOutputTensor(handle, (void*)image_output, RocalTensorLayout::ROCAL_NCHW, RocalTensorOutputType::ROCAL_FP32, 0.485 * 255,0.456 * 255,0.406 * 255, 0.229 * 255,0.224 * 255,0.225 * 255, false);                
                // rocalGetImageLabels(handle,(void*)labels_output);

                c1 = high_resolution_clock::now();
                output_tensor_list = rocalGetOutputTensors(handle);
                c2 = high_resolution_clock::now();
                auto dur = duration_cast<microseconds>( c2 - c1 ).count();
                acc_time += (double)dur;
                c1 = high_resolution_clock::now();
                output_tensor_list->at(0)->copy_data(d_image_output);
                c2 = high_resolution_clock::now();
                dur = duration_cast<microseconds>( c2 - c1 ).count();
                memcpy_acc_time += (double)dur;
                
                c1 = high_resolution_clock::now();
                labels = rocalGetImageLabels(handle);
                memcpy(d_labels_output, labels->at(0)->buffer(), sizeof(int) * labels->size());
                c2 = high_resolution_clock::now();
                dur = duration_cast<microseconds>( c2 - c1 ).count();
                metadata_acc_time += (double)dur;

            // }
            // counter += inputBatchSize;
            // std::cerr << "Inside while loop: " << counter << std::endl;
        }
        auto rocal_timing = rocalGetTimingInfo(handle);
        std::cout << "Load     time "<< (double)rocal_timing.load_time/1000000 << std::endl;
        std::cout << "Decode   time "<< (double)rocal_timing.decode_time/1000000 << std::endl;
        std::cout << "Process  time "<< (double)rocal_timing.process_time/1000000 << std::endl;
        std::cout << "Get ring buffer wait time "<< (double)rocal_timing.transfer_time/1000000 << std::endl;
        std::cout << "rocalGetOutputTensors time "<< (double)acc_time/1000000 << std::endl;
        std::cout << "Image buffer memcpy time "<< (double)memcpy_acc_time/1000000 << std::endl;
        std::cout << "Labels Transfer time "<< (double)metadata_acc_time/1000000 << std::endl;
        std::cout << "Wait if empty time "<< (double)rocal_timing.wait_if_empty_time/1000000 << std::endl;
        std::cout << "Wait if full time "<< (double)rocal_timing.wait_if_full_time/1000000 << std::endl;
        // std::cout << "Circular buffer Wait if empty time "<< (double)rocal_timing.cb_wait_if_empty_time/1000000 << std::endl;
        // std::cout << "Circular buffer Wait if full time "<< (double)rocal_timing.cb_wait_if_full_time/1000000 << std::endl;
        rocalResetLoaders(handle);
        // if(1) {
        //     int val_counter = 0;
        //     while (!rocalIsEmpty(handle1))
        //     {
        //         if(rocalRun(handle1) != 0)
        //             break;

        //         if(processing_device)
        //         {
        //             rocalCopyToOutputTensor(handle1, (void*)d_image_output1, RocalTensorLayout::ROCAL_NCHW, RocalTensorOutputType::ROCAL_FP32, 0.485 * 255,0.456 * 255,0.406 * 255, 0.229 * 255,0.224 * 255,0.225 * 255, false);
        //             rocalGetImageLabels(handle1,(void*)d_val_labels_output);
        //         }
        //         else
        //         {
        //             int val_labels_output[inputBatchSize];
        //             rocalCopyToOutputTensor(handle1, (void*)image_output1, RocalTensorLayout::ROCAL_NCHW, RocalTensorOutputType::ROCAL_FP32, 0.485 * 255,0.456 * 255,0.406 * 255, 0.229 * 255,0.224 * 255,0.225 * 255, false);
        //             rocalGetImageLabels(handle1,(void*)val_labels_output);
        //         }
        //         val_counter += inputBatchSize;
        //         // std::cerr << "Inside while loop: " << counter << std::endl;
        //     }
        //     auto rocal_timing1 = rocalGetTimingInfo(handle1);
        //     std::cout << "Load     time "<< rocal_timing1.load_time/1000000 << std::endl;
        //     std::cout << "Decode   time "<< rocal_timing1.decode_time/1000000 << std::endl;
        //     std::cout << "Process  time "<< rocal_timing1.process_time/1000000 << std::endl;
        //     std::cout << "Transfer time "<< rocal_timing1.transfer_time/1000000 << std::endl;
        //     // std::cout << "Wait if empty time "<< rocal_timing1.wait_if_empty_time << std::endl;
        //     // std::cout << "Wait if full time "<< rocal_timing1.wait_if_full_time << std::endl;
        //     rocalResetLoaders(handle1);
        // }
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>( t2 - t1 ).count();
    std::cout << ">>>>> "<< "Total Elapsed Time " << dur/1000000/60 << " minutes " << dur/1000000%60 << " seconds " << std::endl;
    rocalRelease(handle);
    // rocalRelease(handle1);
    // free(image_output);
    // free(image_output1);
    // if(processing_device)
    // {
    hipFree(d_image_output);
    // hipFree(d_image_output1);
    hipFree(d_labels_output);
    // hipFree(d_val_labels_output);
    // }
    return 0;
}