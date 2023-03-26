from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
from amd.rocal.plugin.pytorch import ROCALNumpyIterator

from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import os


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
    if(sys.argv[2] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    batch_size = int(sys.argv[3])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)

    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads,device_id=device_id, seed=random_seed, rocal_cpu=_rali_cpu)
    local_rank = 0
    world_size = 1
    rali_cpu= True
    rali_device = 'cpu' if rali_cpu else 'gpu'
    decoder_device = 'cpu' if rali_cpu else 'mixed'
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0

    # with pipe:
    #     jpegs, labels = fn.readers.file(file_root=data_path)
    #     images = fn.decoders.image(jpegs,file_root=data_path, output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=False)
    #     brightend_images = fn.brightness(images)
    #     # brightend_images2 = fn.brightness(brightend_images)

    #     pipe.set_outputs(brightend_images)

    # pipe.build()
    # imageIterator = ROCALClassificationIterator(pipe)
    # cnt = 0
    # for i , it in enumerate(imageIterator):
    #     print("************************************** i *************************************",i)
    #     for img in it[0]:
    #         print(img.shape)
    #         cnt = cnt + 1
    #         draw_patches(img, cnt, "cpu")

    print("*********************************************************************")

    pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=_rali_cpu)

    with pipeline:
        numpy_reader_output = fn.readers.numpy(file_root=data_path, is_output=True, shard_id=local_rank, num_shards=world_size)
        pipeline.set_outputs(numpy_reader_output)

    pipeline.build()
    numpyIteratorPipeline = ROCALNumpyIterator(pipeline)
    cnt = 0
    for epoch in range(3):
        print("+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++",epoch)
        for i , it in enumerate(numpyIteratorPipeline):
            print(it.shape)
            print("************************************** i *************************************",i)
            # for img in it[0]:
                # print(img.shape)
                # cnt = cnt + 1
                # draw_patches(img, cnt, "cpu")
        numpyIteratorPipeline.reset()
    print("*********************************************************************")
    exit(0)

if __name__ == '__main__':
    main()
