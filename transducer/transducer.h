#pragma once

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
    bool useCuda);

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
    bool useCuda);

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
    bool useCuda);
