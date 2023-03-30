# Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import rocal_pybind as b
from amd.rocal.pipeline import Pipeline
import amd.rocal.types as types

def file(*inputs, file_root, bytes_per_sample_hint=0, file_list='', initial_fill='', lazy_init='', num_shards=1,
                 pad_last_batch=False, prefetch_queue_depth=1, preserve=False, random_shuffle=False, read_ahead=False,
                 seed=-1, shard_id=0, shuffle_after_epoch=False, skip_cached_images=False, stick_to_shard=False, tensor_init_bytes=1048576, device=None):

    Pipeline._current_pipeline._reader = "labelReader"
    #Output
    labels = []
    kwargs_pybind = {"source_path": file_root}
    label_reader_meta_data = b.labelReader(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (label_reader_meta_data, labels)

def coco(*inputs,file_root='', annotations_file='', bytes_per_sample_hint=0, dump_meta_files=False, dump_meta_files_path='', file_list='', initial_fill=1024,  lazy_init=False, ltrb=False, masks=False, meta_files_path='', num_shards=1, pad_last_batch=False, prefetch_queue_depth=1,
                 preserve=False, random_shuffle=False, ratio=False, read_ahead=False,
                 save_img_ids=False, seed=-1, shard_id=0, shuffle_after_epoch=False, size_threshold=0.1, is_box_encoder=False, is_box_iou_matcher=False,
                 skip_cached_images=False, skip_empty=False, stick_to_shard=False, tensor_init_bytes=1048576):

    Pipeline._current_pipeline._reader = "COCOReader"
    #Output
    labels = []
    bboxes = []
    kwargs_pybind = {"source_path": annotations_file, "is_output":True, "is_box_encoder":is_box_encoder, "is_box_iou_matcher":is_box_iou_matcher }
    b.setSeed(seed)
    meta_data = b.COCOReader(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (meta_data, labels, bboxes)

def numpy(*inputs,file_root='', bytes_per_sample_hint=0, file_list='', initial_fill=1024,  lazy_init=False, num_shards=1, pad_last_batch=False, prefetch_queue_depth=1,
                 preserve=False, random_shuffle=False, ratio=False, read_ahead=False,
                 save_img_ids=False, seed=-1, shard_id=0, shuffle_after_epoch=False,
                 skip_cached_images=False, skip_empty=False, stick_to_shard=False, tensor_init_bytes=1048576, is_output=True):

    Pipeline._current_pipeline._reader = "NumpyReader"
    #Output
    kwargs_pybind = {"source_path": file_root, "is_output": False,"shuffle": random_shuffle, "loop": False, "decode_size_policy":types.MAX_SIZE,"shard_id": shard_id, "shard_count": num_shards}
    numpy_reader_output = b.NumpyReaderSourceShard(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (numpy_reader_output)