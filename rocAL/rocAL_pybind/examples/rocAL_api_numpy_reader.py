from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random

from amd.rocal.pipeline import Pipeline
from amd.rocal.plugin.pytorch import ROCALNumpyIterator
import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import os
import numpy as np

def main():
    if  len(sys.argv) < 3:
        print ('Please pass numpy_folder cpu/gpu batch_size')
        exit(0)
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/NUMPY_READER/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    data_path1 = sys.argv[1]
    data_path2 = sys.argv[2]
    if(sys.argv[3] == "cpu"):
        rocal_cpu = True
    else:
        rocal_cpu = False
    batch_size = int(sys.argv[4])
    num_threads = 8
    device_id = 0
    local_rank = 0
    world_size = 1
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)

    data_files_list = []
    for file in os.listdir(data_path1):
        data_files_list.append(os.path.join(data_path1, file))

    label_files_list = []
    for file in os.listdir(data_path2):
        label_files_list.append(os.path.join(data_path2, file))

    import time
    start = time.time()
    pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=rocal_cpu, prefetch_queue_depth=2)

    with pipeline:
        numpy_reader_output = fn.readers.numpy(file_root=data_path1, shard_id=local_rank, num_shards=world_size)
        numpy_reader_output1 = fn.readers.numpy(file_root=data_path2, shard_id=local_rank, num_shards=world_size)
        data_output = fn.set_layout(numpy_reader_output, output_layout=types.NCDHW)
        label_output = fn.set_layout(numpy_reader_output1, output_layout=types.NCDHW)
        [roi_start, roi_end] = fn.random_object_bbox(label_output, format="start_end", k_largest=3)
        anchor = fn.roi_random_crop(label_output, roi_start=roi_start, roi_end=roi_end, crop_shape=(1, 128, 128, 128), remove_dim=0)
        data_sliced_output = fn.slice(data_output, anchor=anchor, shape=(128,128,128), output_layout=types.NCDHW, output_dtype=types.FLOAT)
        label_sliced_output = fn.slice(label_output, anchor=anchor, shape=(128,128,128), output_layout=types.NCDHW, output_dtype=types.UINT8)       
        hflip = fn.random.coin_flip(probability=0.33)
        vflip = fn.random.coin_flip(probability=0.33)
        dflip = fn.random.coin_flip(probability=0.33)
        data_flip_output = fn.flip(data_sliced_output, horizontal=hflip, vertical=vflip, depth=dflip, output_layout=types.NCDHW, output_dtype=types.FLOAT)
        label_flip_output = fn.flip(label_sliced_output, horizontal=hflip, vertical=vflip, depth=dflip, output_layout=types.NCDHW, output_dtype=types.UINT8)
        brightness = fn.random.uniform(range=[0.7, 1.3])
        add_brightness = fn.random.coin_flip(probability=0.1)
        brightness_output = fn.brightness(data_flip_output, brightness=brightness, brightness_shift=0.0, conditional_execution=add_brightness, output_layout=types.NCDHW, output_dtype=types.FLOAT)
        add_noise = fn.random.coin_flip(probability=0.5)
        std_dev = fn.random.uniform(range=[0.0, 0.1])
        noise_output = fn.gaussian_noise(brightness_output, mean=0.0, std_dev=std_dev, conditional_execution=add_noise, output_layout=types.NCDHW, output_dtype=types.FLOAT)
        pipeline.set_outputs(noise_output, label_flip_output)

    pipeline.build()
    
    numpyIteratorPipeline = ROCALNumpyIterator(pipeline, device='cpu' if rocal_cpu else 'gpu')
    print(len(numpyIteratorPipeline))
    cnt = 0
    for epoch in range(1):
        print("+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++",epoch)
        for i , it in enumerate(numpyIteratorPipeline):
            print(i, it[0].shape, it[1].shape)
            for j in range(batch_size):
                print(it[0][j].cpu().numpy().shape, it[1][j].cpu().numpy().shape)
                cnt += 1
            print("************************************** i *************************************",i)
        numpyIteratorPipeline.reset()
    print("*********************************************************************")
    print(f'Took {time.time() - start} seconds')

if __name__ == '__main__':
    main()
