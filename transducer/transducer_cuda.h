#pragma once

#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) \
  cuda::cudaCheck(err, __FILE__, __LINE__)

namespace cuda {

void cudaCheck(cudaError_t err, const char* file, int line);

void forward(
    const float* emissions,
    const float* predictions,
    float* costs,
    float* alphas,
    const float* logNorms,
    const int* labels,
    const int* inputLengths,
    const int* labelLengths,
    int batchSize,
    int maxInputLength,
    int maxLabelLength,
    int alphabetSize,
    int blank);

void backward(
    const float* emissions,
    const float* predictions,
    float* egrads,
    float* pgrads,
    float* lngrads,
    const float* alphas,
    const float* logNorms,
    const int* labels,
    const int* inputLengths,
    const int* labelLengths,
    int batchSize,
    int maxInputLength,
    int maxLabelLength,
    int alphabetSize,
    int blank);

void viterbi(
    const float* emissions,
    const float* predictions,
    const float* logNorms,
    int* labels,
    const int* inputLengths,
    const int* labelLengths,
    int batchSize,
    int maxInputLength,
    int maxLabelLength,
    int alphabetSize,
    int blank);

} // namespace cuda
