from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random

from amd.rocal.pipeline import Pipeline
from amd.rocal.plugin.pytorch import ROCALNumpyIterator
import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import os, glob

def load_data(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data

def get_data_split(path: str):
    imgs = load_data(path, "data-*.npy")
    lbls = load_data(path, "label-*.npy")
    assert len(imgs) == len(lbls), f"Found {len(imgs)} volumes but {len(lbls)} corresponding masks"
    return imgs, lbls

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
    data_path = sys.argv[1]
    data_path1 = sys.argv[2]
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
    x_train, y_train = get_data_split(data_path)
    x_val, y_val = get_data_split(data_path1)

    import time
    start = time.time()
    pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=rocal_cpu, prefetch_queue_depth=6)

    with pipeline:
        numpy_reader_output = fn.readers.numpy(file_root=data_path, files=x_train, shard_id=local_rank, num_shards=world_size, random_shuffle=True, seed=random_seed+local_rank)
        label_output = fn.readers.numpy(file_root=data_path, files=y_train, shard_id=local_rank, num_shards=world_size, random_shuffle=True, seed=random_seed+local_rank)
        data_output = fn.set_layout(numpy_reader_output, output_layout=types.NHWC)
        pipeline.set_outputs(data_output, label_output)

    pipeline.build()

    pipeline1 = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=rocal_cpu, prefetch_queue_depth=6)

    with pipeline1:
        numpy_reader_output = fn.readers.numpy(file_root=data_path, files=x_val, shard_id=local_rank, num_shards=world_size, seed=random_seed+local_rank)
        label_output = fn.readers.numpy(file_root=data_path, files=y_val, shard_id=local_rank, num_shards=world_size, seed=random_seed+local_rank)
        data_output = fn.set_layout(numpy_reader_output, output_layout=types.NHWC)
        pipeline1.set_outputs(data_output, label_output)

    pipeline1.build()
    
    numpyIteratorPipeline = ROCALNumpyIterator(pipeline, device='cpu' if rocal_cpu else 'gpu')
    print(len(numpyIteratorPipeline))
    valNumpyIteratorPipeline = ROCALNumpyIterator(pipeline1, device='cpu' if rocal_cpu else 'gpu')
    print(len(valNumpyIteratorPipeline))
    cnt = 0
    for epoch in range(2):
        print("+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++",epoch)
        for i , it in enumerate(numpyIteratorPipeline):
            print(i, it[0].shape, it[1].shape)
            for j in range(batch_size):
                print(it[0][j].cpu().numpy().shape, it[1][j].cpu().numpy().shape)
                cnt += 1
            print("************************************** i *************************************",i)
        numpyIteratorPipeline.reset()
        for i , it in enumerate(valNumpyIteratorPipeline):
            print(i, it[0].shape, it[1].shape)
            for j in range(batch_size):
                print(it[0][j].cpu().numpy().shape, it[1][j].cpu().numpy().shape)
                cnt += 1
            print("************************************** i *************************************",i)
        valNumpyIteratorPipeline.reset()
    print("*********************************************************************")
    print(f'Took {time.time() - start} seconds')

if __name__ == '__main__':
    main()
