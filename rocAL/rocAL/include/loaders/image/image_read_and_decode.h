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

#pragma once
#include <dirent.h>
#include <vector>
#include <memory>
#include "commons.h"
#include "turbo_jpeg_decoder.h"
#include "reader_factory.h"
#include "timing_debug.h"
#include "loader_module.h"
#include "parameter_random_crop_decoder.h"
#if _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#endif
#include <math.h>

#define PIXELCHECK(pixel)            (pixel < (float) 0) ? ((float) 0) : ((pixel < (float) 255) ? pixel : ((float) 255))

typedef struct
{
    unsigned int width;
    unsigned int height;
} ImagePatch, *ImagePatchPtr;

typedef struct
{
    int x, y;
    int roiWidth, roiHeight;

} ImageROI, *ImageROIPtr;

typedef struct
{
    unsigned int nStride;
    unsigned int cStride;
    unsigned int hStride;
    unsigned int wStride;
} Strides;

typedef struct
{
    unsigned int n, c, h, w;
    Strides strides;
} Desc, *DescPtr;

const __m128 xmm_p0 = _mm_set1_ps(0.0f);
const __m128i xmm_pxMask00To03 = _mm_setr_epi8(0, 0x80, 0x80, 0x80, 1, 0x80, 0x80, 0x80, 2, 0x80, 0x80, 0x80, 3, 0x80, 0x80, 0x80);
const __m128i xmm_pxMask04To07 = _mm_setr_epi8(4, 0x80, 0x80, 0x80, 5, 0x80, 0x80, 0x80, 6, 0x80, 0x80, 0x80, 7, 0x80, 0x80, 0x80);
const __m128i xmm_pxMask08To11 = _mm_setr_epi8(8, 0x80, 0x80, 0x80, 9, 0x80, 0x80, 0x80, 10, 0x80, 0x80, 0x80, 11, 0x80, 0x80, 0x80);
const __m128i xmm_pxMask12To15 = _mm_setr_epi8(12, 0x80, 0x80, 0x80, 13, 0x80, 0x80, 0x80, 14, 0x80, 0x80, 0x80, 15, 0x80, 0x80, 0x80);

inline void resize_store_pkd3(unsigned char *dstPtr, __m128 *p)
{
    __m128i px[7];
    __m128i pxMask = _mm_setr_epi8(0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 13, 14, 15);
    __m128i pxZero = _mm_setzero_si128();

    px[4] = _mm_cvtps_epi32(p[0]);    /* convert to int32 for R01-04 */
    px[5] = _mm_cvtps_epi32(p[4]);    /* convert to int32 for G01-04 */
    px[6] = _mm_cvtps_epi32(p[8]);    /* convert to int32 for B01-04 */
    px[4] = _mm_packus_epi32(px[4], px[5]);    /* pack pixels 0-7 as R01-04|G01-04 */
    px[5] = _mm_packus_epi32(px[6], pxZero);    /* pack pixels 8-15 as B01-04|X01-04 */
    px[0] = _mm_packus_epi16(px[4], px[5]);    /* pack pixels 0-15 as [R01|R02|R03|R04|G01|G02|G03|G04|B01|B02|B03|B04|00|00|00|00] */
    px[4] = _mm_cvtps_epi32(p[1]);    /* convert to int32 for R05-08 */
    px[5] = _mm_cvtps_epi32(p[5]);    /* convert to int32 for G05-08 */
    px[6] = _mm_cvtps_epi32(p[9]);    /* convert to int32 for B05-08 */
    px[4] = _mm_packus_epi32(px[4], px[5]);    /* pack pixels 0-7 as R05-08|G05-08 */
    px[5] = _mm_packus_epi32(px[6], pxZero);    /* pack pixels 8-15 as B05-08|X01-04 */
    px[1] = _mm_packus_epi16(px[4], px[5]);    /* pack pixels 0-15 as [R05|R06|R07|R08|G05|G06|G07|G08|B05|B06|B07|B08|00|00|00|00] */
    px[4] = _mm_cvtps_epi32(p[2]);    /* convert to int32 for R09-12 */
    px[5] = _mm_cvtps_epi32(p[6]);    /* convert to int32 for G09-12 */
    px[6] = _mm_cvtps_epi32(p[10]);    /* convert to int32 for B09-12 */
    px[4] = _mm_packus_epi32(px[4], px[5]);    /* pack pixels 0-7 as R09-12|G09-12 */
    px[5] = _mm_packus_epi32(px[6], pxZero);    /* pack pixels 8-15 as B09-12|X01-04 */
    px[2] = _mm_packus_epi16(px[4], px[5]);    /* pack pixels 0-15 as [R09|R10|R11|R12|G09|G10|G11|G12|B09|B10|B11|B12|00|00|00|00] */
    px[4] = _mm_cvtps_epi32(p[3]);    /* convert to int32 for R13-16 */
    px[5] = _mm_cvtps_epi32(p[7]);    /* convert to int32 for G13-16 */
    px[6] = _mm_cvtps_epi32(p[11]);    /* convert to int32 for B13-16 */
    px[4] = _mm_packus_epi32(px[4], px[5]);    /* pack pixels 0-7 as R13-16|G13-16 */
    px[5] = _mm_packus_epi32(px[6], pxZero);    /* pack pixels 8-15 as B13-16|X01-04 */
    px[3] = _mm_packus_epi16(px[4], px[5]);    /* pack pixels 0-15 as [R13|R14|R15|R16|G13|G14|G15|G16|B13|B14|B15|B16|00|00|00|00] */
    px[0] = _mm_shuffle_epi8(px[0], pxMask);    /* shuffle to get [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|00|00|00|00] */
    px[1] = _mm_shuffle_epi8(px[1], pxMask);    /* shuffle to get [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|00|00|00|00] */
    px[2] = _mm_shuffle_epi8(px[2], pxMask);    /* shuffle to get [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|00|00|00|00] */
    px[3] = _mm_shuffle_epi8(px[3], pxMask);    /* shuffle to get [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|00|00|00|00] */
    _mm_storeu_si128((__m128i *)dstPtr, px[0]);           /* store [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 12), px[1]);    /* store [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 24), px[2]);    /* store [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 36), px[3]);    /* store [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|00|00|00|00] */
}
inline void set_zeros(__m128 *pVecs, int numVecs)
{
    for(int i = 0; i < numVecs; i++)
        pVecs[i] = xmm_p0;
}

inline void resize_load(unsigned char *srcPtr, __m128 *p)
{
    __m128i px = _mm_loadu_si128((__m128i *)srcPtr);    /* load pixels 0-15 */
    p[0] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px, xmm_pxMask00To03));    /* pixels 0-3 */
    p[1] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px, xmm_pxMask04To07));    /* pixels 4-7 */
    p[2] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px, xmm_pxMask08To11));    /* pixels 8-11 */
    p[3] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px, xmm_pxMask12To15));    /* pixels 12-15 */
}

inline void compute_resize_src_loc(int dstLocation, float scale, int &srcLoc, float &weight, float offset = 0, unsigned int srcStride = 1)
{
    float srcLocationFloat = ((float) dstLocation) * scale + offset;
    int srcLocation = (int) std::ceil(srcLocationFloat);
    weight = srcLocation - srcLocationFloat;
    srcLoc = srcLocation * srcStride;
}

inline void compute_row_coefficients(float radius, float scale, float size, float weight, float *coeffs, unsigned int srcStride = 1)
{
    float sum = 0;
    weight = weight - radius;
    for(int k = 0; k < size; k++)
    {
        float coeff;
        coeff = 1 - std::abs((weight + k) * scale);
        coeffs[k] = coeff < 0 ? 0 : coeff;
        sum += coeffs[k];
    }
    if(sum)
    {
        sum = 1 / sum;
        for(int k = 0; k < size; k++)
            coeffs[k] = coeffs[k] * sum;
    }
}
inline void compute_col_coefficients(float radius, float scale, float size, float weight, float *coeffs, unsigned int srcStride = 1)
{
    float sum = 0;
    weight = weight - radius;

    // The coefficients are computed for 4 dst locations and stored consecutively for ease of access
    for(int k = 0, kPos = 0; k < size; k++, kPos += 4)
    {
        float coeff;
        coeff = 1 - std::fabs((weight + k) * scale);
        coeffs[kPos] = coeff < 0 ? 0 : coeff;
        sum += coeffs[kPos];
    }
    if(sum)
    {
        sum = 1 / sum;
        for(int k = 0, kPos = 0; k < size; k++, kPos += 4)
            coeffs[kPos] = coeffs[kPos] * sum;
    }
}

inline void compute_separable_vertical_resample(unsigned char *inputPtr, float *outputPtr, DescPtr inputDescPtr, DescPtr outputDescPtr,
                                                ImagePatch inputImgSize, ImagePatch outputImgSize, int *index, float *coeffs, int size)
{

    static constexpr int maxNumLanes = 16;                                  // Maximum number of pixels that can be present in a vector for U8 type
    static constexpr int loadLanes = maxNumLanes / sizeof(unsigned char);
    static constexpr int storeLanes = maxNumLanes / sizeof(float);
    static constexpr int numLanes = std::max(loadLanes, storeLanes);        // No of pixels that can be present in a vector wrt data type
    static constexpr int numVecs = numLanes * sizeof(float) / maxNumLanes; // No of float vectors required to process numLanes pixels

    int inputHeightLimit = inputImgSize.height - 1;
    int outPixelsPerIter = 4;
    
    std::cerr << "Before creating uchar buffer in computing vertical resample: " << size << "\n";
    unsigned char *inRowPtr[size];
    std::cerr << "Before for loop in computing vertical resample\n";
    for (int outLocRow = 0; outLocRow < (int)outputImgSize.height; outLocRow++)
    {
        __m128 pCoeff[size];
        int k0 = outLocRow * size;
        float *outRowPtr = outputPtr + outLocRow * outputDescPtr->strides.hStride;

        // Determine the input row pointers and coefficients to be used for interpolation
        for (int k = 0; k < size; k++)
        {
            int inLocRow = index[outLocRow] + k;
            inLocRow = std::min(std::max(inLocRow, 0), inputHeightLimit);
            inRowPtr[k] = inputPtr + inLocRow * inputDescPtr->strides.hStride;
            pCoeff[k] = _mm_set1_ps(coeffs[k0 + k]);    // Each row is associated with a single coeff
        }
        int bufferLength = inputImgSize.width * inputDescPtr->strides.wStride;
        int alignedLength = bufferLength &~ (numLanes-1);
        int outLocCol = 0;

        // Load the input pixels from size rows
        // Multiply input vec from each row with it's correspondig coefficient
        // Add the results from size rows to obtain the pixels of an output row
        for (; outLocCol + numLanes <= alignedLength; outLocCol += numLanes)
        {
            __m128 pTemp[numVecs];
            set_zeros(pTemp, numVecs);
            for (int k = 0; k < size; k++)
            {
                __m128 pInput[numVecs];
                resize_load(inRowPtr[k] + outLocCol, pInput);   // Load numLanes input pixels from each row
                for (int v = 0; v < numVecs; v++)
                    pTemp[v] = _mm_fmadd_ps(pInput[v], pCoeff[k], pTemp[v]);
            }
            for(int vec = 0, outStoreStride = 0; vec < numVecs; vec++, outStoreStride += outPixelsPerIter)     // Since 4 output pixels are stored per iteration
                _mm_storeu_ps(outRowPtr + outLocCol + outStoreStride, pTemp[vec]);
        }

        for (; outLocCol < bufferLength; outLocCol++)
        {
            float temp = 0;
            for (int k = 0; k < size; k++)
                temp += (inRowPtr[k][outLocCol] * coeffs[k0 + k]);
            outRowPtr[outLocCol] = temp;
        }
    }
    std::cerr << "vertical resample computed\n";
}

inline void compute_separable_horizontal_resample(float *inputPtr, unsigned char *outputPtr, DescPtr inputDescPtr, DescPtr outputDescPtr,
                        ImagePatch inputImgSize, ImagePatch outputImgSize, int *index, float *coeffs, int size, float radius)
{
    static constexpr int maxNumLanes = 16;                                   // Maximum number of pixels that can be present in a vector
    static constexpr int numLanes = maxNumLanes / sizeof(unsigned char);                 // No of pixels that can be present in a vector wrt data type
    static constexpr int numVecs = numLanes * sizeof(float) / maxNumLanes;  // No of float vectors required to process numLanes pixels
    int numOutPixels, filterKernelStride;
    numOutPixels = filterKernelStride = 4;

    int inputWidthLimit = (inputImgSize.width - 1) * inputDescPtr->strides.wStride;

    for (int outLocRow = 0; outLocRow < (int)outputImgSize.height; outLocRow++)
    {
        unsigned char *outRowPtrR = outputPtr + outLocRow * outputDescPtr->strides.hStride;
        unsigned char *outRowPtrG = outRowPtrR + outputDescPtr->strides.cStride;
        unsigned char *outRowPtrB = outRowPtrG + outputDescPtr->strides.cStride;
        float *inRowPtr = inputPtr + outLocRow * inputDescPtr->strides.hStride;
        int bufferLength = outputImgSize.width;
        int alignedLength = bufferLength &~ (numLanes-1);
        int outLocCol = 0;

        // Load size consecutive pixels from a location in the row
        // Multiply with corresponding coeffs and add together to obtain the output pixel
        for (; outLocCol + numLanes <= alignedLength; outLocCol += numLanes)
        {
            __m128 pOutputChannel[numVecs * 3];
            set_zeros(pOutputChannel, numVecs * 3);
            __m128 *pOutputR = pOutputChannel;
            __m128 *pOutputG = pOutputChannel + numVecs;
            __m128 *pOutputB = pOutputChannel + (numVecs * 2);
            for(int vec = 0, x = outLocCol; vec < numVecs; vec++, x += numOutPixels)   // 4 dst pixels processed per iteration
            {
                int coeffIdx = (x * size);
                for(int k = 0, kStrided = 0; k < size; k ++, kStrided = k * 3)
                {
                    __m128 pInput[numOutPixels];
                    __m128 pCoeffs = _mm_loadu_ps(&(coeffs[coeffIdx + (k * numOutPixels)]));
                    for (int l = 0; l < numOutPixels; l++)
                    {
                        int srcx = index[x + l] + kStrided;
                        srcx = std::min(std::max(srcx, 0), inputWidthLimit);
                        pInput[l] = _mm_loadu_ps(&inRowPtr[srcx]);
                    }

                    // Perform transpose operation to arrange input pixels by R,G and B separately in each vector
                    _MM_TRANSPOSE4_PS(pInput[0], pInput[1], pInput[2], pInput[3]);
                    pOutputR[vec] = _mm_fmadd_ps(pCoeffs, pInput[0], pOutputR[vec]);
                    pOutputG[vec] = _mm_fmadd_ps(pCoeffs, pInput[1], pOutputG[vec]);
                    pOutputB[vec] = _mm_fmadd_ps(pCoeffs, pInput[2], pOutputB[vec]);
                }
            }

            int xStride = outLocCol * outputDescPtr->strides.wStride;
            resize_store_pkd3(outRowPtrR + xStride, pOutputChannel);
        }
        int k0 = 0;
        for (; outLocCol < (int)outputImgSize.width; outLocCol++)
        {
            int x0 = index[outLocCol];
            k0 = outLocCol % 4 == 0 ? outLocCol * size : k0 + 1;  // Since coeffs are stored in continuously for 4 dst locations
            float sumR, sumG, sumB;
            sumR = sumG = sumB = 0;
            for (int k = 0; k < size; k++)
            {
                int srcx = x0 + (k * 3);
                srcx = std::min(std::max(srcx, 0), inputWidthLimit);
                int kPos = (k * 4);      // Since coeffs are stored in continuously for 4 dst locations
                sumR += (coeffs[k0 + kPos] * inRowPtr[srcx]);
                sumG += (coeffs[k0 + kPos] * inRowPtr[srcx + 1]);
                sumB += (coeffs[k0 + kPos] * inRowPtr[srcx + 2]);
            }
            int xStride = outLocCol * outputDescPtr->strides.wStride;
            *(outRowPtrR + xStride) = PIXELCHECK(sumR);
            *(outRowPtrG + xStride) = PIXELCHECK(sumG);
            *(outRowPtrB + xStride) = PIXELCHECK(sumB);
        }
    }
}

typedef struct
{
    unsigned int width;
    unsigned int height;
} imSize;

struct ResizetensorLocalData
{
    uint32_t nbatchSize;
    imSize *srcDimensions;
    imSize maxSrcDimensions;
    imSize *dstDimensions;
    imSize maxDstDimensions;
    DescPtr srcDescPtr, dstDescPtr, tempDescPtr;
    ImageROIPtr roiTensorPtrSrc;
    ImagePatchPtr dstImgSize;
    Desc srcDesc, dstDesc, tempDesc;
};

void resize_tensor_host(std::vector<std::vector<unsigned char>>& srcPtr,
                        DescPtr srcDescPtr,
                        unsigned char *dstPtr,
                        DescPtr dstDescPtr,
                        float * tempPtr,
                        DescPtr tempDescPtr,
                        ImagePatchPtr dstImgSize,
                        ImageROIPtr roiTensorPtrSrc,
                        size_t num_threads);

/**
 * Compute the scaled value of <tt>dimension</tt> using the given scaling
 * factor.  This macro performs the integer equivalent of <tt>ceil(dimension *
 * scalingFactor)</tt>.
 */
#define TJSCALED(dimension, scalingFactor) \
  ((dimension * scalingFactor.num + scalingFactor.denom - 1) / \
   scalingFactor.denom)

class ImageReadAndDecode
{
public:
    ImageReadAndDecode();
    ~ImageReadAndDecode();
    size_t count();
    void reset();
    void create(ReaderConfig reader_config, DecoderConfig decoder_config, int batch_size, int device_id=0);
    void set_bbox_vector(std::vector<std::vector <float>> bbox_coords) { _bbox_coords = bbox_coords;};
    void set_random_bbox_data_reader(std::shared_ptr<RandomBBoxCrop_MetaDataReader> randombboxcrop_meta_data_reader);
    std::vector<std::vector <float>> get_batch_random_bbox_crop_coords();
    void set_batch_random_bbox_crop_coords(std::vector<std::vector <float>> batch_crop_coords);

    //! Loads a decompressed batch of images into the buffer indicated by buff
    /// \param buff User's buffer provided to be filled with decoded image samples
    /// \param names User's buffer provided to be filled with name of the images decoded
    /// \param max_decoded_width User's buffer maximum width per decoded image. User expects the decoder to downscale the image if image's original width is bigger than max_width
    /// \param max_decoded_height user's buffer maximum height per decoded image. User expects the decoder to downscale the image if image's original height is bigger than max_height
    /// \param roi_width is set by the load() function tp the width of the region that decoded image is located. It's less than max_width and is either equal to the original image width if original image width is smaller than max_width or downscaled if necessary to fit the max_width criterion.
    /// \param roi_height  is set by the load() function tp the width of the region that decoded image is located.It's less than max_height and is either equal to the original image height if original image height is smaller than max_height or downscaled if necessary to fit the max_height criterion.
    /// \param output_color_format defines what color format user expects decoder to decode images into if capable of doing so supported is
    LoaderModuleStatus load(
            unsigned char* buff,
            std::vector<std::string>& names,
            const size_t  max_decoded_width,
            const size_t max_decoded_height,
            std::vector<uint32_t> &roi_width,
            std::vector<uint32_t> &roi_height,
            std::vector<uint32_t> &actual_width,
            std::vector<uint32_t> &actual_height,
            RocalColorFormat output_color_format,
            bool decoder_keep_original=false);

    //! returns timing info or other status information
    Timing timing();

private:
    std::vector<std::shared_ptr<Decoder>> _decoder;
    std::shared_ptr<Reader> _reader;
    std::vector<std::vector<unsigned char>> _compressed_buff;
    std::vector<std::vector<unsigned char>> _decompressed_buff;
    std::vector<size_t> _actual_read_size;
    std::vector<std::string> _image_names;
    std::vector<size_t> _compressed_image_size;
    std::vector<unsigned char*> _decompressed_buff_ptrs;
    std::vector<size_t> _actual_decoded_width;
    std::vector<size_t> _actual_decoded_height;
    std::vector<size_t> _original_width;
    std::vector<size_t> _original_height;
    static const size_t MAX_COMPRESSED_SIZE = 1*1024*1024; // 1 Meg
    TimingDBG _file_load_time, _decode_time;
    size_t _batch_size, _shard_count, _num_threads;
    DecoderConfig _decoder_config;
    bool decoder_keep_original;
    bool _resize_with_decode;
    unsigned _max_decoded_width, _max_decoded_height;
    unsigned _resize_width, _resize_height;
    float* _tempFloatmem;
    ResizetensorLocalData *data;
    std::vector<std::vector <float>> _bbox_coords, _crop_coords_batch;
    std::shared_ptr<RandomBBoxCrop_MetaDataReader> _randombboxcrop_meta_data_reader = nullptr;
    pCropCord _CropCord;
    RocalRandomCropDecParam *_random_crop_dec_param = nullptr;
};

