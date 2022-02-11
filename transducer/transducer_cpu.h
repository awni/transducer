#pragma once

namespace cpu {

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
    int blank);

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
    int blank);

} // namespace cpu
