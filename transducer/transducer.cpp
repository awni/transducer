#include "transducer.h"
#include "transducer_cpu.h"

#if defined(_CUDA_)
#include "transducer_cuda.h"
#endif

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
    int blank,
    bool useCuda) {
  if (useCuda) {
#if defined(_CUDA_)
    cuda::forward(
        emissions,
        predictions,
        costs,
        alphas,
        logNorms,
        labels,
        inputLengths,
        labelLengths,
        batchSize,
        maxInputLength,
        maxLabelLength,
        alphabetSize,
        blank);
#else
    throw std::invalid_argument("Transducer not built with CUDA.");
#endif
  } else {
    cpu::forward(
        emissions,
        predictions,
        costs,
        alphas,
        logNorms,
        labels,
        inputLengths,
        labelLengths,
        batchSize,
        maxInputLength,
        maxLabelLength,
        alphabetSize,
        blank);
  }
}

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
    int blank,
    bool useCuda) {
  if (useCuda) {
#if defined(_CUDA_)
    cuda::backward(
        emissions,
        predictions,
        egrads,
        pgrads,
        lngrads,
        alphas,
        logNorms,
        labels,
        inputLengths,
        labelLengths,
        batchSize,
        maxInputLength,
        maxLabelLength,
        alphabetSize,
        blank);
#else
    throw std::invalid_argument("Transducer not built with CUDA.");
#endif
  } else {
    cpu::backward(
        emissions,
        predictions,
        egrads,
        pgrads,
        lngrads,
        alphas,
        logNorms,
        labels,
        inputLengths,
        labelLengths,
        batchSize,
        maxInputLength,
        maxLabelLength,
        alphabetSize,
        blank);
  }
}

void viterbi(
    const float* emissions,
    const float* predictions,
    int* labels,
    const int* inputLengths,
    const int* labelLengths,
    int batchSize,
    int maxInputLength,
    int maxLabelLength,
    int alphabetSize,
    int blank,
    bool useCuda) {
  if (useCuda) {
#if defined(_CUDA_)
    cuda::viterbi(
        emissions,
        predictions,
        labels,
        inputLengths,
        labelLengths,
        batchSize,
        maxInputLength,
        maxLabelLength,
        alphabetSize,
        blank);
#else
    throw std::invalid_argument("Transducer not built with CUDA.");
#endif
  } else {
    cpu::viterbi(
        emissions,
        predictions,
        labels,
        inputLengths,
        labelLengths,
        batchSize,
        maxInputLength,
        maxLabelLength,
        alphabetSize,
        blank);
  }
}
