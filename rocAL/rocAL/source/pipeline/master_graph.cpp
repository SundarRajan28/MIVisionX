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
#if !ENABLE_HIP
#include <CL/cl.h>
#endif
#include <vx_ext_amd.h>
#include <VX/vx_types.h>
#include <cstring>
#include <sched.h>
#include <half/half.hpp>
#include "master_graph.h"
#include "parameter_factory.h"
#include "ocl_setup.h"
#include "log.h"
#include "meta_data_reader_factory.h"
#include "meta_data_graph_factory.h"
#include "randombboxcrop_meta_data_reader_factory.h"
#include "node_copy.h"

using half_float::half;

#if ENABLE_SIMD
#if _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#endif
#endif

#if ENABLE_HIP
#include <rocal_hip_kernels.h>
#endif

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char* string)
{
    size_t len = strnlen(string, MAX_STRING_LENGTH);
    if (len > 0) {
        printf("%s", string);
        if (string[len - 1] != '\n')
            printf("\n");
        fflush(stdout);
    }
}


auto get_ago_affinity_info = []
    (RocalAffinity rocal_affinity,
     int cpu_id,
     int gpu_id)
{
    AgoTargetAffinityInfo affinity;
    switch(rocal_affinity) {
        case RocalAffinity::GPU:
            affinity.device_type =  AGO_TARGET_AFFINITY_GPU;
            affinity.device_info = (gpu_id >=0 && gpu_id <=9)? gpu_id : 0;
            break;
        case RocalAffinity::CPU:
            affinity.device_type = AGO_TARGET_AFFINITY_CPU;
            affinity.device_info = (cpu_id >=0 && cpu_id <=9)? cpu_id : 0;
            break;
        default:
            throw std::invalid_argument("Unsupported affinity");
    }
    return affinity;
};

MasterGraph::~MasterGraph()
{
    release();
}

MasterGraph::MasterGraph(size_t batch_size, RocalAffinity affinity, int gpu_id, size_t prefetch_queue_depth, RocalTensorDataType output_tensor_data_type):
        _ring_buffer(prefetch_queue_depth),
        _graph(nullptr),
        _affinity(affinity),
        _gpu_id(gpu_id),
        _convert_time("Conversion Time", DBG_TIMING),
        _process_time("Process Time", DBG_TIMING),
        _bencode_time("BoxEncoder Time", DBG_TIMING),
        _output_routine_time("Output_routine Time", DBG_TIMING),
        _user_batch_size(batch_size),
#if ENABLE_HIP
        _mem_type ((_affinity == RocalAffinity::GPU) ? RocalMemType::HIP : RocalMemType::HOST),
#else
        _mem_type ((_affinity == RocalAffinity::GPU) ? RocalMemType::OCL : RocalMemType::HOST),
#endif
        _first_run(true),
        _processing(false),
        _prefetch_queue_depth(prefetch_queue_depth),
        _out_data_type(output_tensor_data_type),
#if ENABLE_HIP
        _box_encoder_gpu(nullptr),
#endif
        _rb_block_if_empty_time("Ring Buffer Block IF Empty Time"),
        _rb_block_if_full_time("Ring Buffer Block IF Full Time")
{
    try {
        vx_status status;
        vxRegisterLogCallback(NULL, log_callback, vx_false_e);
        _context = vxCreateContext();
        vxRegisterLogCallback(_context, log_callback, vx_false_e);
        auto vx_affinity = get_ago_affinity_info(_affinity, 0, gpu_id);
        if ((status = vxGetStatus((vx_reference) _context)) != VX_SUCCESS)
            THROW("vxCreateContext failed" + TOSTR(status))

        if(affinity == RocalAffinity::GPU)
        {
#if !ENABLE_HIP
            if (_mem_type == RocalMemType::OCL){
                cl_context _cl_context = nullptr;
                cl_device_id _cl_device_id = nullptr;
                get_device_and_context(gpu_id, &_cl_context, &_cl_device_id, CL_DEVICE_TYPE_GPU);
                if((status = vxSetContextAttribute(_context,
                        VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT,
                        &_cl_context, sizeof(cl_context)) != VX_SUCCESS))
                    THROW("vxSetContextAttribute for CL_CONTEXT failed " + TOSTR(status))
            }
#else
            if (_mem_type == RocalMemType::HIP) {
                hipError_t err = hipInit(0);
                if (err != hipSuccess) {
                    THROW("ERROR: hipInit(0) => %d (failed)" + TOSTR(err));
                }
                // initialize HIP device for rocAL
                int hip_num_devices = -1;
                err = hipGetDeviceCount(&hip_num_devices);
                if (err != hipSuccess) {
                    THROW("ERROR: hipGetDeviceCount() => %d (failed)" + TOSTR(err));
                }
                //set the device for context if specified.
                if (gpu_id < hip_num_devices) {
                    int hipDevice = gpu_id;
                    if((status = vxSetContextAttribute(_context,
                            VX_CONTEXT_ATTRIBUTE_AMD_HIP_DEVICE,
                            &hipDevice, sizeof(hipDevice)) != VX_SUCCESS))
                        THROW("vxSetContextAttribute for hipDevice(%d) failed " + TOSTR(hipDevice) + TOSTR(status))
                }else
                    THROW("ERROR: HIP Device(%d) out of range" + TOSTR(gpu_id));

            }
#endif
        }

        // Setting attribute to run on CPU or GPU should be called before load kernel module
        if ((status = vxSetContextAttribute(_context,
                                            VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY,
                                            &vx_affinity,
                                            sizeof(vx_affinity))) != VX_SUCCESS)
            THROW("vxSetContextAttribute for AMD_AFFINITY failed " + TOSTR(status))

        // loading OpenVX RPP modules
        if ((status = vxLoadKernels(_context, "vx_rpp")) != VX_SUCCESS)
            THROW("Cannot load vx_rpp extension (vx_rpp), vxLoadKernels failed " + TOSTR(status))
        else
            LOG("vx_rpp module loaded successfully")
// #ifdef ROCAL_VIDEO
//         // loading video decoder modules
//         if ((status = vxLoadKernels(_context, "vx_amd_media")) != VX_SUCCESS)
//             WRN("Cannot load vx_amd_media extension, video decode functionality will not be available")
//         else
//             LOG("vx_amd_media module loaded")
// #endif
        if(_affinity == RocalAffinity::GPU) {
#if ENABLE_HIP
            _device.init_hip(_context);
#else
            _device.init_ocl(_context);
#endif
        }
    }
    catch(const std::exception& e)
    {
        release();
        throw;
    }
}

MasterGraph::Status
MasterGraph::run()
{
    if(!_processing)// The user should not call the run function before the build() is called or while reset() is happening
        return MasterGraph::Status::NOT_RUNNING;

    if(no_more_processed_data()) {
        return MasterGraph::Status::NO_MORE_DATA;
    }

    _rb_block_if_empty_time.start();
    _ring_buffer.block_if_empty();// wait here if the user thread (caller of this function) is faster in consuming the processed images compare to th output routine in producing them
    _rb_block_if_empty_time.end();

    if(_first_run)
    {
        // calling run pops the processed images that have been used by user, when user calls run() for the first time
        // they've not used anything yet, so we don't pop a batch from the _ring_buffer
        _first_run = false;
    } else {
        _ring_buffer.pop(); // Pop previously used output images and metadata from the ring buffer
    }

    // If the last batch of processed imaged has been just popped from the ring_buffer it means user has previously consumed all the processed images.
    // User should check using the IsEmpty() API and not call run() or copy() API when there is no more data. run() will return MasterGraph::Status::NO_MORE_DATA flag to notify it.
    if(no_more_processed_data()) {
        return MasterGraph::Status::NO_MORE_DATA;
    }

    decrease_image_count();

    return MasterGraph::Status::OK;
}

void
MasterGraph::decrease_image_count()
{
    if(!_loop)
        _remaining_count -= (_is_sequence_reader_output ? _sequence_batch_size : _user_batch_size);
}

void
MasterGraph::create_single_graph()
{
    // Actual graph creating and calls into adding nodes to graph is deferred and is happening here to enable potential future optimizations
    _graph = std::make_shared<Graph>(_context, _affinity, 0, _gpu_id);
    for(auto& node: _nodes)
    {
        // Any tensor not yet created can be created as virtual tensor
        for(auto& tensor: node->output())
            if(tensor->info().type() == rocalTensorInfo::Type::UNKNOWN)
            {
                tensor->create_virtual(_context, _graph->get());
                _internal_tensors.push_back(tensor);
            }
        node->create(_graph);
    }
    _graph->verify();
}

MasterGraph::Status
MasterGraph::build()
{
    if(_internal_tensor_list.empty())
        THROW("No output tensors are there, cannot create the pipeline")

#if ENABLE_HIP || ENABLE_OPENCL
    _ring_buffer.init(_mem_type, (void *)_device.resources(), _internal_tensor_list.data_size(), _internal_tensor_list.size());
#else
    _ring_buffer.init(_mem_type, nullptr, _internal_tensor_list.data_size(), _internal_tensor_list.size());
#endif
    if (_is_box_encoder) _ring_buffer.initBoxEncoderMetaData(_mem_type, _user_batch_size*_num_anchors*4*sizeof(float), _user_batch_size*_num_anchors*sizeof(int));
    // _output_tensor_list = _internal_tensor_list;
    create_single_graph();
    start_processing();
    return Status::OK;
}

rocalTensor *
MasterGraph::create_loader_output_tensor(const rocalTensorInfo &info) {
    /*
    *   NOTE: Output tensor for a source node needs to be created as a regular (non-virtual) tensor
    */
    auto output = new rocalTensor(info);
    if(output->create_from_handle(_context) != 0)
        THROW("Creating output tensor for loader failed");

    _internal_tensors.push_back(output);

    return output;
}

rocalTensor *
MasterGraph::create_tensor(const rocalTensorInfo &info, bool is_output) {
    auto *output = new rocalTensor(info);
    // if the tensor is not an output tensor, the tensor creation is deferred and later it'll be created as a virtual tensor
    if(is_output) {
        if (output->create_from_handle(_context) != 0)
            THROW("Cannot create the tensor from handle")
        _internal_tensor_list.push_back(output);
        _output_tensor_list.push_back(new rocalTensor(info));   // Creating a replica of the output tensor to be returned to the user
    }

    return output;
}

void
MasterGraph::set_output(rocalTensor* output_tensor)
{
    if(output_tensor->is_handle_set() == false)
    {
        if (output_tensor->create_from_handle(_context) != 0)
                THROW("Cannot create the tensor from handle")

        _internal_tensor_list.push_back(output_tensor);
        _output_tensor_list.push_back(new rocalTensor(output_tensor->info()));  // Creating a replica of the output tensor to be returned to the user
    }
    else
    {
        // Decoder case only
        auto actual_output = create_tensor(output_tensor->info(), true);
        add_node<CopyNode>({output_tensor}, {actual_output});
    }
}

void MasterGraph::release()
{
    LOG("MasterGraph release ...")
    stop_processing();
    _nodes.clear();
    _root_nodes.clear();
    _meta_data_nodes.clear();
    _tensor_map.clear();
    _ring_buffer.release_gpu_res();
    //shut_down loader:: required for releasing any allocated resourses
    _loader_module->shut_down();
    // release all openvx resources.
    vx_status status;
    for(auto& tensor: _internal_tensors)
        delete tensor;  // It will call the vxReleaseTensor internally in the destructor
    _internal_tensor_list.release(); // It will call the vxReleaseTensor internally in the destructor for each tensor in the list
    _output_tensor_list.release();   // It will call the vxReleaseTensor internally in the destructor for each tensor in the list
    for(auto tensor_list: _metadata_output_tensor_list)
        tensor_list->release(); // It will call the vxReleaseTensor internally in the destructor for each tensor in the list

    if(_graph != nullptr)
        _graph->release();
    if(_context && (status = vxReleaseContext(&_context)) != VX_SUCCESS)
        LOG ("Failed to call vxReleaseContext " + TOSTR(status))

    _augmented_meta_data = nullptr;
    _meta_data_graph = nullptr;
    _meta_data_reader = nullptr;
}

MasterGraph::Status
MasterGraph::update_node_parameters()
{
    // Randomize random parameters
    ParameterFactory::instance()->renew_parameters();

    // Apply renewed parameters to VX parameters used in augmentation
    for(auto& node: _nodes)
        node->update_parameters();

    return Status::OK;
}

std::vector<uint32_t>
MasterGraph::output_resize_width()
{
    std::vector<uint32_t> resize_width_vector;
    resize_width_vector = _resize_width.back();
    _resize_width.pop_back();
    return resize_width_vector;
}

std::vector<uint32_t>
MasterGraph::output_resize_height()
{
    std::vector<uint32_t> resize_height_vector;
    resize_height_vector = _resize_height.back();
    _resize_height.pop_back();
    return resize_height_vector;
}

void
MasterGraph::sequence_start_frame_number(std::vector<size_t> &sequence_start_framenum)
{
    sequence_start_framenum = _sequence_start_framenum_vec.back();
    _sequence_start_framenum_vec.pop_back();
}

void
MasterGraph::sequence_frame_timestamps(std::vector<std::vector<float>> &sequence_frame_timestamp)
{
    sequence_frame_timestamp = _sequence_frame_timestamps_vec.back();
    _sequence_frame_timestamps_vec.pop_back();
}

MasterGraph::Status
MasterGraph::reset()
{
    // stop the internal processing thread so that the
    _processing = false;
    _ring_buffer.unblock_writer();
    if(_output_thread.joinable())
        _output_thread.join();
    _ring_buffer.reset();
    _sequence_start_framenum_vec.clear();
    _sequence_frame_timestamps_vec.clear();
    // clearing meta ring buffer
    // if random_bbox meta reader is used: read again to get different crops
    if (_randombboxcrop_meta_data_reader != nullptr)
        _randombboxcrop_meta_data_reader->release();
    // resetting loader module to start from the beginning of the media and clear it's internal state/buffers
    _loader_module->reset();
    // restart processing of the images
    _first_run = true;
    _output_routine_finished_processing = false;
    _resize_width.clear();
    _resize_height.clear();
    start_processing();
    return Status::OK;
}

size_t
MasterGraph::remaining_count()
{
    return (_remaining_count >= 0) ? _remaining_count:0;
}

RocalMemType
MasterGraph::mem_type()
{
    return _mem_type;
}

Timing
MasterGraph::timing()
{
    Timing t = _loader_module->timing();
    t.image_process_time += _process_time.get_timing();
    t.copy_to_output += _convert_time.get_timing();
    t.bb_process_time += _bencode_time.get_timing();
    t.image_output_routine_time += _output_routine_time.get_timing();
    t.wait_if_empty_time += _ring_buffer._rb_block_if_empty_time.get_timing();
    t.wait_if_full_time += _ring_buffer._rb_block_if_full_time.get_timing();
    return t;
}

rocalTensorList *
MasterGraph::get_output_tensors()
{
    _convert_time.start();
    std::vector<void*> output_ptr = _ring_buffer.get_read_buffers();
    _convert_time.end();
    for(unsigned i = 0; i < _internal_tensor_list.size(); i++)
        _output_tensor_list[i]->set_mem_handle(output_ptr[i]);
    
    // _output_tensor_list[0]->set_mem_handle(_ring_buffer.get_read_buffers()[0]);
    return &_output_tensor_list;
}


ImageNameBatch& operator+=(ImageNameBatch& dest, const ImageNameBatch& src)
{
    dest.insert(dest.end(), src.cbegin(), src.cend());
    return dest;
}

void MasterGraph::output_routine()
{
    INFO("Output routine started with "+TOSTR(_remaining_count) + " to load");
    try {
        while (_processing)
        {
            ImageNameBatch full_batch_image_names = {};
            pMetaDataBatch full_batch_meta_data = nullptr;
            pMetaDataBatch augmented_batch_meta_data = nullptr;
            if (_loader_module->remaining_count() < (_is_sequence_reader_output ? _sequence_batch_size : _user_batch_size))
            {
                // If the internal process routine ,output_routine(), has finished processing all the images, and last
                // processed images stored in the _ring_buffer will be consumed by the user when it calls the run() func
                notify_user_thread();
                // the following call is required in case the ring buffer is waiting for more data to be loaded and there is no more data to process.
                _ring_buffer.release_if_empty();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            _rb_block_if_full_time.start();
            // _ring_buffer.get_write_buffers() is blocking and blocks here until user uses processed image by calling run() and frees space in the ring_buffer
            auto write_buffers = _ring_buffer.get_write_buffers();
            _rb_block_if_full_time.end();

            // Swap handles on the input tensor, so that new tensor is loaded to be processed
            auto load_ret = _loader_module->load_next();
            if (load_ret != LoaderModuleStatus::OK)
                THROW("Loader module failed to load next batch of images, status " + TOSTR(load_ret))
            if (!_processing)
                break;
            auto this_cycle_names =  _loader_module->get_id();
            auto decode_image_info = _loader_module->get_decode_image_info();
            auto crop_image_info = _loader_module->get_crop_image_info();

            if(this_cycle_names.size() != _user_batch_size)
                WRN("Internal problem: names count "+ TOSTR(this_cycle_names.size()))

            // meta_data lookup is done before _meta_data_graph->process() is called to have the new meta_data ready for processing
            if (_meta_data_reader)
                _meta_data_reader->lookup(this_cycle_names);

            full_batch_image_names += this_cycle_names;

            if (!_processing)
                break;

            // Swap handles on the output tensor, so that new processed tensor will be written to the a new buffer
            for (size_t idx = 0; idx < _internal_tensor_list.size(); idx++)
            {
                _internal_tensor_list[idx]->swap_handle(write_buffers[idx]);
            }

            if (!_processing)
                break;

            for(auto node: _nodes)
            {
                if(node->_is_ssd)
                {
                    node->set_meta_data(_augmented_meta_data);
                }
            }

            update_node_parameters();
            if(_augmented_meta_data)
            {
                if (_meta_data_graph)
                {
                    if(_is_random_bbox_crop)
                    {
                        _meta_data_graph->update_random_bbox_meta_data(_augmented_meta_data, decode_image_info, crop_image_info);
                    }
                    else
                    {
                        _meta_data_graph->update_meta_data(_augmented_meta_data, decode_image_info, _is_segmentation);
                    }
                    _meta_data_graph->process(_augmented_meta_data, _is_segmentation);
                }
                if (full_batch_meta_data)
                    full_batch_meta_data->concatenate(_augmented_meta_data);
                else
                    full_batch_meta_data = _augmented_meta_data->clone();
            }

            // get roi width and height of output image -> For Mask!
            // std::vector<uint32_t> temp_width_arr;
            // std::vector<uint32_t> temp_height_arr;
            // for (unsigned int i = 0; i < _user_batch_size; i++)
            // {
            //     temp_width_arr.push_back(_internal_tensor_list.front()->info().get_roi()[i].x2);
            //     temp_height_arr.push_back(_internal_tensor_list.front()->info().get_roi()[i].y2);
            // }
            // _resize_width.insert(_resize_width.begin(), temp_width_arr);
            // _resize_height.insert(_resize_height.begin(), temp_height_arr);
            
            _process_time.start();
            _graph->process();
            _process_time.end();
            
            _bencode_time.start();
            if(_is_box_encoder )
            {
#if ENABLE_HIP
                if(_mem_type == RocalMemType::HIP) {
                    // get bbox encoder read buffers
                    auto bbox_encode_write_buffers = _ring_buffer.get_box_encode_write_buffers();
                    if (_box_encoder_gpu) _box_encoder_gpu->Run(full_batch_meta_data, (float *)bbox_encode_write_buffers.first, (int *)bbox_encode_write_buffers.second);
                    //_meta_data_graph->update_box_encoder_meta_data_gpu(_anchors_gpu_buf, num_anchors, full_batch_meta_data, _criteria, _offset, _scale, _means, _stds);
                } else
#endif
                    _meta_data_graph->update_box_encoder_meta_data(&_anchors, full_batch_meta_data, _criteria, _offset, _scale, _means, _stds);
            }
            _bencode_time.end();
            _ring_buffer.set_meta_data(full_batch_image_names, full_batch_meta_data, _is_segmentation);
            _ring_buffer.push(); // Image data and metadata is now stored in output the ring_buffer, increases it's level by 1
        }
    }
    catch (const std::exception &e)
    {
        ERR("Exception thrown in the process routine: " + STR(e.what()) + STR("\n"));
        _processing = false;
        _ring_buffer.release_all_blocked_calls();
    }
}

void MasterGraph::start_processing()
{
    _processing = true;
    _remaining_count = _loader_module->remaining_count();
    _output_thread = std::thread(&MasterGraph::output_routine, this);
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#else
//  Changing thread scheduling policy and it's priority does not help on latest Ubuntu builds
//  and needs tweaking the Linux security settings , can be turned on for experimentation
#if 0
    struct sched_param params;
    params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    auto thread = _output_thread.native_handle();
    auto ret = pthread_setschedparam(thread, SCHED_FIFO, &params);
    if (ret != 0)
        WRN("Unsuccessful in setting thread realtime priority for process thread err = "+STR(std::strerror(ret)))
#endif
#endif
}

void MasterGraph::stop_processing()
{
    _processing = false;
    _ring_buffer.unblock_reader();
    _ring_buffer.unblock_writer();
    if(_output_thread.joinable())
        _output_thread.join();
}

std::vector<rocalTensorList *> MasterGraph::create_mxnet_label_reader(const char *source_path, bool is_output)
{
    if( _meta_data_reader)
        THROW("A metadata reader has already been created")
    MetaDataConfig config(MetaDataType::Label, MetaDataReaderType::MXNET_META_DATA_READER, source_path);
    _meta_data_graph = create_meta_data_graph(config);
    _meta_data_reader = create_meta_data_reader(config);
    _meta_data_reader->init(config);
    _meta_data_reader->read_all(source_path);
    unsigned num_of_dims = 1;
    std::vector<size_t> dims;
    dims.resize(num_of_dims);
    dims.at(0) = 1; // Number of labels per file
    auto default_labels_info  = rocalTensorInfo(std::vector<size_t>(std::move(dims)),
                                _mem_type,
                                RocalTensorDataType::INT32);
    default_labels_info.set_metadata();
    _meta_data_buffer_size.emplace_back(_user_batch_size * sizeof(vx_int32));

    for(unsigned i = 0; i < _user_batch_size; i++)
    {
        auto info = default_labels_info;
        auto tensor = new rocalTensor(info);
        _labels_tensor_list.push_back(tensor);
    }
    _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);
    _ring_buffer.init_metadata(RocalMemType::HOST, _meta_data_buffer_size, _meta_data_buffer_size.size());
    if(is_output)
    {
        if (_augmented_meta_data)
            THROW("Metadata output already defined, there can only be a single output for metadata augmentation")
        else
            _augmented_meta_data = _meta_data_reader->get_output();
    }
    return _metadata_output_tensor_list;
}

std::vector<rocalTensorList *> MasterGraph::create_cifar10_label_reader(const char *source_path, const char *file_prefix)
{
    if( _meta_data_reader)
        THROW("A metadata reader has already been created")
    MetaDataConfig config(MetaDataType::Label, MetaDataReaderType::CIFAR10_META_DATA_READER, source_path, std::map<std::string, std::string>(), file_prefix);
    _meta_data_reader = create_meta_data_reader(config);
    _meta_data_reader->init(config);
    _meta_data_reader->read_all(source_path);
    unsigned num_of_dims = 1;
    std::vector<size_t> dims;
    dims.resize(num_of_dims);
    dims.at(0) = 1; // Number of labels per file
    auto default_labels_info  = rocalTensorInfo(std::vector<size_t>(std::move(dims)),
                                _mem_type,
                                RocalTensorDataType::INT32);
    default_labels_info.set_metadata();
    _meta_data_buffer_size.emplace_back(_user_batch_size * sizeof(vx_int32));

    for(unsigned i = 0; i < _user_batch_size; i++)
    {
        auto info = default_labels_info;
        auto tensor = new rocalTensor(info);
        _labels_tensor_list.push_back(tensor);
    }
    _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);
    
    
    _ring_buffer.init_metadata(RocalMemType::HOST, _meta_data_buffer_size, _meta_data_buffer_size.size());
    if (_augmented_meta_data)
        THROW("Metadata can only have a single output")
    else
        _augmented_meta_data = _meta_data_reader->get_output();
    return _metadata_output_tensor_list;
}

std::vector<rocalTensorList *> MasterGraph::create_video_label_reader(const char *source_path, MetaDataReaderType reader_type, unsigned sequence_length, unsigned frame_step, unsigned frame_stride, bool file_list_frame_num)
{
    if( _meta_data_reader)
        THROW("A metadata reader has already been created")
    MetaDataConfig config(MetaDataType::Label, reader_type, source_path, std::map<std::string, std::string>(), std::string(), false, sequence_length, frame_step, frame_stride);
    _meta_data_reader = create_meta_data_reader(config);
    _meta_data_reader->init(config);
    if(!file_list_frame_num)
    {
        _meta_data_reader->set_timestamp_mode();
    }

    unsigned num_of_dims = 1;
    std::vector<size_t> dims;
    dims.resize(num_of_dims);
    dims.at(0) = 1; // Number of labels per file
    auto default_labels_info  = rocalTensorInfo(std::vector<size_t>(std::move(dims)),
                                 _mem_type,
                                 RocalTensorDataType::INT32);
    default_labels_info.set_metadata();
    _meta_data_buffer_size.emplace_back(_user_batch_size * sizeof(vx_int32));

    for(unsigned i = 0; i < _user_batch_size; i++)
    {
        auto info = default_labels_info;
        auto tensor = new rocalTensor(info);
        _labels_tensor_list.push_back(tensor);
    }
    _ring_buffer.init_metadata(RocalMemType::HOST, _meta_data_buffer_size, _meta_data_buffer_size.size());

    _meta_data_reader->read_all(source_path);
    if (_augmented_meta_data)
        THROW("Metadata can only have a single output")
    else
        _augmented_meta_data = _meta_data_reader->get_output();
    _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);

    return _metadata_output_tensor_list;
}

std::vector<rocalTensorList *> MasterGraph::create_coco_meta_data_reader(const char *source_path, bool is_output, bool mask, MetaDataReaderType reader_type, MetaDataType label_type, bool is_box_encoder)
{
    if(_meta_data_reader)
        THROW("A metadata reader has already been created")
    if(mask)
        _is_segmentation = true;
    MetaDataConfig config(label_type, reader_type, source_path, std::map<std::string, std::string>(), std::string(), mask);
    _meta_data_graph = create_meta_data_graph(config);
    _meta_data_reader = create_meta_data_reader(config);
    _meta_data_reader->init(config);
    _meta_data_reader->read_all(source_path);
    unsigned num_of_dims = 1;
    std::vector<size_t> dims;
    dims.resize(num_of_dims);
    dims.at(0) = is_box_encoder ? MAX_NUM_ANCHORS : MAX_OBJECTS;
    auto default_labels_info  = rocalTensorInfo(dims,
                                        _mem_type,
                                        RocalTensorDataType::INT32);
    default_labels_info.set_metadata();
    _meta_data_buffer_size.emplace_back(dims.at(0) * _user_batch_size * sizeof(vx_int32)); // TODO - replace with data size from info

    num_of_dims = 2;
    dims.resize(num_of_dims);
    dims.at(0) = is_box_encoder ? MAX_NUM_ANCHORS : MAX_OBJECTS;
    dims.at(1) = BBOX_COUNT;
    auto default_bbox_info  = rocalTensorInfo(dims,
                                        _mem_type,
                                        RocalTensorDataType::FP32);
    default_bbox_info.set_metadata();
    _meta_data_buffer_size.emplace_back(dims.at(0) * dims.at(1)  * _user_batch_size * sizeof(vx_float32)); // TODO - replace with data size from info

    rocalTensorInfo default_mask_info;
    if(mask)
    {
        num_of_dims = 2;
        dims.resize(num_of_dims);
        dims.at(0) = MAX_MASK_BUFFER;
        dims.at(1) = 1;
        default_mask_info  = rocalTensorInfo(dims,
                                            _mem_type,
                                            RocalTensorDataType::FP32);
        default_mask_info.set_metadata();
        _meta_data_buffer_size.emplace_back(dims.at(0) * dims.at(1)  * _user_batch_size * sizeof(vx_float32)); // TODO - replace with data size from info  
    }

    for(unsigned i = 0; i < _user_batch_size; i++)
    {
        auto labels_info = default_labels_info;
        auto bbox_info = default_bbox_info;
        _labels_tensor_list.push_back(new rocalTensor(labels_info));
        _bbox_tensor_list.push_back(new rocalTensor(bbox_info));
        if(mask)
        {
            auto mask_info = default_mask_info;
            _mask_tensor_list.push_back(new rocalTensor(mask_info));
        }
    }
    _ring_buffer.init_metadata(RocalMemType::HOST, _meta_data_buffer_size, _meta_data_buffer_size.size());
    if(is_output)
    {
        if (_augmented_meta_data)
            THROW("Metadata output already defined, there can only be a single output for metadata augmentation")
        else
            _augmented_meta_data = _meta_data_reader->get_output();
    }
    _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);
    _metadata_output_tensor_list.emplace_back(&_bbox_tensor_list);
    if(mask)
        _metadata_output_tensor_list.emplace_back(&_mask_tensor_list);

    return _metadata_output_tensor_list;
}

std::vector<rocalTensorList *> MasterGraph::create_caffe2_lmdb_record_meta_data_reader(const char *source_path, MetaDataReaderType reader_type , MetaDataType label_type)
{
    if( _meta_data_reader)
        THROW("A metadata reader has already been created")
    MetaDataConfig config(label_type, reader_type, source_path);
    _meta_data_graph = create_meta_data_graph(config);
    _meta_data_reader = create_meta_data_reader(config);
    _meta_data_reader->init(config);
    _meta_data_reader->read_all(source_path);
    if(reader_type == MetaDataReaderType::CAFFE2_META_DATA_READER)
    {
        unsigned num_of_dims = 1;
        std::vector<size_t> dims;
        dims.resize(num_of_dims);
        dims.at(0) = 1; // Number of labels per file
        auto default_labels_info  = rocalTensorInfo(std::vector<size_t>(std::move(dims)),
                                    _mem_type,
                                    RocalTensorDataType::INT32);
        default_labels_info.set_metadata();
        _meta_data_buffer_size.emplace_back(_user_batch_size * sizeof(vx_int32));

        for(unsigned i = 0; i < _user_batch_size; i++)
        {
            auto info = default_labels_info;
            auto tensor = new rocalTensor(info);
            _labels_tensor_list.push_back(tensor);
        }
        _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);
    }
    else if(reader_type == MetaDataReaderType::CAFFE2_DETECTION_META_DATA_READER)
    {
        unsigned num_of_dims = 1;
        std::vector<size_t> dims;
        dims.resize(num_of_dims);
        dims.at(0) = MAX_OBJECTS;
        auto default_labels_info  = rocalTensorInfo(std::vector<size_t>(std::move(dims)),
                                            _mem_type,
                                            RocalTensorDataType::INT32);
        default_labels_info.set_metadata();

        num_of_dims = 2;
        dims.resize(num_of_dims);
        dims.at(0) = MAX_OBJECTS;
        dims.at(1) = BBOX_COUNT;
        auto default_bbox_info  = rocalTensorInfo(std::vector<size_t>(std::move(dims)),
                                            _mem_type,
                                            RocalTensorDataType::FP32);
        default_bbox_info.set_metadata();
        _meta_data_buffer_size.emplace_back(MAX_OBJECTS * _user_batch_size * sizeof(vx_int32));
        _meta_data_buffer_size.emplace_back(MAX_OBJECTS * BBOX_COUNT  * _user_batch_size * sizeof(vx_float32));

        for(unsigned i = 0; i < _user_batch_size; i++)
        {
            auto labels_info = default_labels_info;
            auto bbox_info = default_bbox_info;
            _labels_tensor_list.push_back(new rocalTensor(labels_info));
            _bbox_tensor_list.push_back(new rocalTensor(bbox_info));
        }
        _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);
        _metadata_output_tensor_list.emplace_back(&_bbox_tensor_list);
    }

    _ring_buffer.init_metadata(RocalMemType::HOST, _meta_data_buffer_size, _meta_data_buffer_size.size());
    if (_augmented_meta_data)
        THROW("Metadata output already defined, there can only be a single output for metadata augmentation")
    else
        _augmented_meta_data = _meta_data_reader->get_output();
    return _metadata_output_tensor_list;
}
std::vector<rocalTensorList *> MasterGraph::create_tf_record_meta_data_reader(const char *source_path, MetaDataReaderType reader_type , MetaDataType label_type, std::map<std::string, std::string> feature_key_map)
{
    if( _meta_data_reader)
        THROW("A metadata reader has already been created")
    MetaDataConfig config(label_type, reader_type, source_path, feature_key_map);
    _meta_data_graph = create_meta_data_graph(config);
    _meta_data_reader = create_meta_data_reader(config);
    _meta_data_reader->init(config);
    _meta_data_reader->read_all(source_path);

    if(reader_type == MetaDataReaderType::TF_META_DATA_READER)
     {
        unsigned num_of_dims = 1;
        std::vector<size_t> dims;
        dims.resize(num_of_dims);
        dims.at(0) = 1; // Number of labels per file
        auto default_labels_info  = rocalTensorInfo(std::vector<size_t>(std::move(dims)),
                                    _mem_type,
                                    RocalTensorDataType::INT32);
        default_labels_info.set_metadata();
        _meta_data_buffer_size.emplace_back(_user_batch_size * sizeof(vx_int32));

        for(unsigned i = 0; i < _user_batch_size; i++)
        {
            auto info = default_labels_info;
            auto tensor = new rocalTensor(info);
            _labels_tensor_list.push_back(tensor);
        }
        _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);
    }
    else if(reader_type == MetaDataReaderType::TF_DETECTION_META_DATA_READER)
    {
        unsigned num_of_dims = 1;
        std::vector<size_t> dims;
        dims.resize(num_of_dims);
        dims.at(0) = MAX_OBJECTS;
        auto default_labels_info  = rocalTensorInfo(std::vector<size_t>(std::move(dims)),
                                            _mem_type,
                                            RocalTensorDataType::INT32);
        default_labels_info.set_metadata();

        num_of_dims = 2;
        dims.resize(num_of_dims);
        dims.at(0) = MAX_OBJECTS;
        dims.at(1) = BBOX_COUNT;
        auto default_bbox_info  = rocalTensorInfo(std::vector<size_t>(std::move(dims)),
                                            _mem_type,
                                            RocalTensorDataType::FP32);
        default_bbox_info.set_metadata();
        _meta_data_buffer_size.emplace_back(MAX_OBJECTS * _user_batch_size * sizeof(vx_int32));
        _meta_data_buffer_size.emplace_back(MAX_OBJECTS * BBOX_COUNT  * _user_batch_size * sizeof(vx_float32));

        for(unsigned i = 0; i < _user_batch_size; i++)
        {
            auto labels_info = default_labels_info;
            auto bbox_info = default_bbox_info;
            _labels_tensor_list.push_back(new rocalTensor(labels_info));
            _bbox_tensor_list.push_back(new rocalTensor(bbox_info));
        }
        _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);
        _metadata_output_tensor_list.emplace_back(&_bbox_tensor_list);
    }

    _ring_buffer.init_metadata(RocalMemType::HOST, _meta_data_buffer_size, _meta_data_buffer_size.size());
    if (_augmented_meta_data)
        THROW("Metadata can only have a single output")
    else
        _augmented_meta_data = _meta_data_reader->get_output();

    return _metadata_output_tensor_list;
}

std::vector<rocalTensorList *> MasterGraph::create_label_reader(const char *source_path, MetaDataReaderType reader_type)
{
    if(_meta_data_reader)
        THROW("A metadata reader has already been created")
    MetaDataConfig config(MetaDataType::Label, reader_type, source_path);
    _meta_data_reader = create_meta_data_reader(config);
    _meta_data_reader->init(config);
    _meta_data_reader->read_all(source_path);

    unsigned num_of_dims = 1;
    std::vector<size_t> dims;
    dims.resize(num_of_dims);
    dims.at(0) = 1; // Number of labels per file
    auto default_labels_info  = rocalTensorInfo(std::vector<size_t>(std::move(dims)),
                                 _mem_type,
                                 RocalTensorDataType::INT32);
    default_labels_info.set_metadata();
    _meta_data_buffer_size.emplace_back(_user_batch_size * sizeof(vx_int32));

    for(unsigned i = 0; i < _user_batch_size; i++)
    {
        auto info = default_labels_info;
        _labels_tensor_list.push_back(new rocalTensor(info));
    }
    _ring_buffer.init_metadata(RocalMemType::HOST, _meta_data_buffer_size, _meta_data_buffer_size.size());
    if (_augmented_meta_data)
        THROW("Metadata can only have a single output")
    else
        _augmented_meta_data = _meta_data_reader->get_output();
    _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);

    return _metadata_output_tensor_list;
}

void MasterGraph::create_randombboxcrop_reader(RandomBBoxCrop_MetaDataReaderType reader_type, RandomBBoxCrop_MetaDataType label_type, bool all_boxes_overlap, bool no_crop, FloatParam* aspect_ratio, bool has_shape, int crop_width, int crop_height, int num_attempts, FloatParam* scaling, int total_num_attempts, int64_t seed)
{
    if( _randombboxcrop_meta_data_reader)
        THROW("A metadata reader has already been created")
    _is_random_bbox_crop = true;
    RandomBBoxCrop_MetaDataConfig config(label_type, reader_type, all_boxes_overlap, no_crop, aspect_ratio, has_shape, crop_width, crop_height, num_attempts, scaling, total_num_attempts, seed);
    _randombboxcrop_meta_data_reader = create_meta_data_reader(config);
    _randombboxcrop_meta_data_reader->set_meta_data(_meta_data_reader);
    if (_random_bbox_crop_cords_data)
        THROW("Metadata can only have a single output")
    else
        _random_bbox_crop_cords_data = _randombboxcrop_meta_data_reader->get_output();
}

void MasterGraph::box_encoder(std::vector<float> &anchors, float criteria, const std::vector<float> &means, const std::vector<float> &stds, bool offset, float scale)
{
    _is_box_encoder = true;
    _num_anchors = anchors.size() / 4;
    std::vector<float> inv_stds = {(float)(1./stds[0]), (float)(1./stds[1]), (float)(1./stds[2]), (float)(1./stds[3])};

#if ENABLE_HIP
    // Intialize gpu box encoder if _mem_type is HIP
    if(_mem_type == RocalMemType::HIP) {
        _box_encoder_gpu = new BoxEncoderGpu(_user_batch_size, anchors, criteria, means, inv_stds, offset, scale, _device.resources()->hip_stream, _device.resources()->dev_prop.canMapHostMemory);
        return;
    }
#endif
    _offset = offset;
    _anchors = anchors;
    _scale = scale;
    _means = means;
    _stds = stds;
}

std::vector<rocalTensorList *> MasterGraph::create_caffe_lmdb_record_meta_data_reader(const char *source_path, MetaDataReaderType reader_type , MetaDataType label_type)
{
    if( _meta_data_reader)
        THROW("A metadata reader has already been created")
    MetaDataConfig config(label_type, reader_type, source_path);
    _meta_data_graph = create_meta_data_graph(config);
    _meta_data_reader = create_meta_data_reader(config);
    _meta_data_reader->init(config);
    _meta_data_reader->read_all(source_path);
    if(reader_type == MetaDataReaderType::CAFFE_META_DATA_READER)
     {
        unsigned num_of_dims = 1;
        std::vector<size_t> dims;
        dims.resize(num_of_dims);
        dims.at(0) = 1; // Number of labels per file
        auto default_labels_info  = rocalTensorInfo(std::vector<size_t>(std::move(dims)),
                                    _mem_type,
                                    RocalTensorDataType::INT32);
        default_labels_info.set_metadata();
        _meta_data_buffer_size.emplace_back(_user_batch_size * sizeof(vx_int32));

        for(unsigned i = 0; i < _user_batch_size; i++)
        {
            auto info = default_labels_info;
            auto tensor = new rocalTensor(info);
            _labels_tensor_list.push_back(tensor);
        }
        _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);
    }
    else if(reader_type == MetaDataReaderType::CAFFE_DETECTION_META_DATA_READER)
    {
        unsigned num_of_dims = 1;
        std::vector<size_t> dims;
        dims.resize(num_of_dims);
        dims.at(0) = MAX_OBJECTS;
        auto default_labels_info  = rocalTensorInfo(std::vector<size_t>(std::move(dims)),
                                            _mem_type,
                                            RocalTensorDataType::INT32);
        default_labels_info.set_metadata();

        num_of_dims = 2;
        dims.resize(num_of_dims);
        dims.at(0) = MAX_OBJECTS;
        dims.at(1) = BBOX_COUNT;
        auto default_bbox_info  = rocalTensorInfo(std::vector<size_t>(std::move(dims)),
                                            _mem_type,
                                            RocalTensorDataType::FP32);
        default_bbox_info.set_metadata();
        _meta_data_buffer_size.emplace_back(MAX_OBJECTS * _user_batch_size * sizeof(vx_int32));
        _meta_data_buffer_size.emplace_back(MAX_OBJECTS * BBOX_COUNT  * _user_batch_size * sizeof(vx_float32));

        for(unsigned i = 0; i < _user_batch_size; i++)
        {
            auto labels_info = default_labels_info;
            auto bbox_info = default_bbox_info;
            _labels_tensor_list.push_back(new rocalTensor(labels_info));
            _bbox_tensor_list.push_back(new rocalTensor(bbox_info));
        }
        _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);
        _metadata_output_tensor_list.emplace_back(&_bbox_tensor_list);
    }

    _ring_buffer.init_metadata(RocalMemType::HOST, _meta_data_buffer_size, _meta_data_buffer_size.size());
    if (_augmented_meta_data)
        THROW("Metadata output already defined, there can only be a single output for metadata augmentation")
    else
        _augmented_meta_data = _meta_data_reader->get_output();
    return _metadata_output_tensor_list;
}
const std::pair<ImageNameBatch,pMetaDataBatch>& MasterGraph::meta_data()
{
    if(_ring_buffer.level() == 0)
        THROW("No meta data has been loaded")
    return _ring_buffer.get_meta_data();
}

rocalTensorList * MasterGraph::labels_meta_data()
{
    if(_ring_buffer.level() == 0)
        THROW("No meta data has been loaded")
    auto meta_data_buffers = (unsigned char *)_ring_buffer.get_meta_read_buffers()[0]; // Get labels buffer from ring buffer
    for(unsigned i = 0; i < _labels_tensor_list.size(); i++)
    {
        _labels_tensor_list[i]->set_mem_handle((void *)meta_data_buffers); // TODO - Need to update according to the metadata
        meta_data_buffers += _labels_tensor_list[i]->info().data_size();
    }

    return &_labels_tensor_list;
}

rocalTensorList * MasterGraph::bbox_labels_meta_data()
{
    if(_ring_buffer.level() == 0)
        THROW("No meta data has been loaded")
    auto meta_data_buffers = (unsigned char *)_ring_buffer.get_meta_read_buffers()[0]; // Get labels buffer from ring buffer
    auto labels_tensor_dims = _ring_buffer.get_meta_data_info().bb_labels_dims();
    for(unsigned i = 0; i < _labels_tensor_list.size(); i++)
    {
        _labels_tensor_list[i]->set_dims(labels_tensor_dims[i]);
        _labels_tensor_list[i]->set_mem_handle((void *)meta_data_buffers);
        meta_data_buffers += _labels_tensor_list[i]->info().data_size();
    }
    return &_labels_tensor_list;
}

rocalTensorList * MasterGraph::bbox_meta_data()
{
    if(_ring_buffer.level() == 0)
        THROW("No meta data has been loaded")
    auto meta_data_buffers = (unsigned char *)_ring_buffer.get_meta_read_buffers()[1]; // Get bbox buffer from ring buffer
    auto bbox_tensor_dims = _ring_buffer.get_meta_data_info().bb_cords_dims();
    for(unsigned i = 0; i < _bbox_tensor_list.size(); i++)
    {
        _bbox_tensor_list[i]->set_dims(bbox_tensor_dims[i]);
        _bbox_tensor_list[i]->set_mem_handle((void *)meta_data_buffers);
        meta_data_buffers += _bbox_tensor_list[i]->info().data_size();
    }

    return &_bbox_tensor_list;
}

size_t MasterGraph::bounding_box_batch_count(int *buf, pMetaDataBatch meta_data_batch)
{
    size_t size = 0;
    for(unsigned i = 0; i < _user_batch_size; i++)
    {
        buf[i] = _is_box_encoder? _num_anchors: meta_data_batch->get_bb_labels_batch()[i].size();
        size += buf[i];
    }
    return size;
}

rocalTensorList * MasterGraph::mask_meta_data()
{
    if(_ring_buffer.level() == 0)
        THROW("No meta data has been loaded")
    auto meta_data_buffers = (unsigned char *)_ring_buffer.get_meta_read_buffers()[2]; // Get bbox buffer from ring buffer
    auto mask_tensor_dims = _ring_buffer.get_meta_data_info().mask_cords_dims();
    for(unsigned i = 0; i < _mask_tensor_list.size(); i++)
    {
        _mask_tensor_list[i]->set_dims(mask_tensor_dims[i]);
        _mask_tensor_list[i]->set_mem_handle((void *)meta_data_buffers);
        meta_data_buffers += _mask_tensor_list[i]->info().data_size();
    }

    return &_mask_tensor_list;
}

ImgSizes& MasterGraph::get_image_sizes()
{
    if(_ring_buffer.level() == 0)
        THROW("No meta data has been loaded")
    return _ring_buffer.get_meta_data().second->get_img_sizes_batch();
}

std::vector<size_t> MasterGraph::tensor_output_byte_size()
{
    return _internal_tensor_list.data_size();
}

void MasterGraph::notify_user_thread()
{
    if(_output_routine_finished_processing)
        return;
    LOG("Output routine finished processing all images, no more image to be processed")
    _output_routine_finished_processing = true;
}

bool MasterGraph::no_more_processed_data()
{
    return (_output_routine_finished_processing && _ring_buffer.empty());
}

std::vector<rocalTensorList *>
MasterGraph::get_bbox_encoded_buffers(size_t num_encoded_boxes)
{
    std::vector<rocalTensorList *> bbox_encoded_output;
    if (_is_box_encoder) {
        if (num_encoded_boxes != _user_batch_size*_num_anchors) {
            THROW("num_encoded_boxes is not correct");
        }
        auto encoded_boxes_and_lables = _ring_buffer.get_box_encode_read_buffers();
        unsigned char *boxes_buf_ptr = (unsigned char *) encoded_boxes_and_lables.first;
        unsigned char *labels_buf_ptr = (unsigned char *) encoded_boxes_and_lables.second;
        auto labels_tensor_dims = _ring_buffer.get_meta_data_info().bb_labels_dims();
        auto bbox_tensor_dims = _ring_buffer.get_meta_data_info().bb_cords_dims();

        if(_bbox_tensor_list.size() != _labels_tensor_list.size())
            THROW("The number of tensors between bbox and bbox_labels do not match")
        for(unsigned i = 0; i < _bbox_tensor_list.size(); i++)
        {
            _labels_tensor_list[i]->set_dims(labels_tensor_dims[i]);
            _bbox_tensor_list[i]->set_dims(bbox_tensor_dims[i]);
            _labels_tensor_list[i]->set_mem_handle((void *)labels_buf_ptr);
            _bbox_tensor_list[i]->set_mem_handle((void *)boxes_buf_ptr);
            labels_buf_ptr += _labels_tensor_list[i]->info().data_size();
            boxes_buf_ptr += _bbox_tensor_list[i]->info().data_size();
        }
        bbox_encoded_output.emplace_back(&_labels_tensor_list);
        bbox_encoded_output.emplace_back(&_bbox_tensor_list);
    }
    return bbox_encoded_output;
}
