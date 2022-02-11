#include "transducer.h"

void forward(
    const float* emissions,
    const float* predictions,
    float* costs,
    float* alphas,
    float* logNorms,
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
    // TODO
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
    // TODO
  } else {
    cpu::backward(
        emissions,
        predictions,
        egrads,
        pgrads,
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
