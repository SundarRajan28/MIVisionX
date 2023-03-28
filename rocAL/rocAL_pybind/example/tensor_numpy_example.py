from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random

from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import os


class ROCALNumpyIterator(object):
    def __init__(self, pipeline, tensor_dtype=types.FLOAT, device="cpu", device_id=0):
        try:
            assert pipeline is not None, "Number of provided pipelines has to be at least 1"
        except Exception as ex:
            print(ex)
        self.loader = pipeline
        self.tensor_dtype = tensor_dtype
        self.device = device
        self.device_id = device_id
        self.out = None
        print("self.device", self.device)
        self.len = self.loader.getRemainingImages()


    def next(self):
        return self.__next__()

    def __next__(self):
        if(self.loader.isEmpty()):
            # timing_info = self.loader.Timing_Info()
            # print("Load     time ::", timing_info.load_time/1000000)
            # print("Decode   time ::", timing_info.decode_time/1000000)
            # print("Process  time ::", timing_info.process_time/1000000)
            # print("Output routine time ::", timing_info.output_routine_time/1000000)
            # print("Transfer time ::", timing_info.transfer_time/1000000)
            raise StopIteration

        if self.loader.rocalRun() != 0:
            raise StopIteration
        else:
            self.output_tensor_list = self.loader.rocalGetOutputTensors()

        #From init
        self.augmentation_count = len(self.output_tensor_list)
        self.batch_size = self.output_tensor_list[0].batch_size()
        self.out = self.output_tensor_list[0].as_array()

        return self.out
        # elif self.tensor_dtype == types.FLOAT16:

    def reset(self):
        self.loader.rocalResetLoaders()

    def __iter__(self):
        return self

    def __len__(self):
        return self.len

    # def __del__(self):
        # b.rocalRelease(self.loader._handle)

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
        numpy_reader_output = fn.readers.numpy(file_root=data_path, shard_id=local_rank, num_shards=world_size)
        pipeline.set_outputs(numpy_reader_output)

    pipeline.build()
    
    numpyIteratorPipeline = ROCALNumpyIterator(pipeline, tensor_dtype=types.UINT8)
    print(len(numpyIteratorPipeline))
    cnt = 0
    for epoch in range(1):
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
