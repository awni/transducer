#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(_OPENMP_)
#include <omp.h>
#endif

#include "transducer_cpu.h"

namespace {
constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

inline float logSumExp(float a, float b) {
  if (a == kNegInf) {
    return b;
  }
  if (b == kNegInf) {
    return a;
  }
  return std::max(a, b) + std::log1p(std::exp(-std::abs(a - b)));
}

inline int idx2(int t, int u, int U) {
  return t * U + u;
}

float forwardSingle(
    const float* emissions,
    const float* predictions,
    float* alphas,
    const float* logNorms,
    const int* labels,
    int blank, int T,
    int U, int maxU, int V) {

  if (T == 0 || U == 0) {
    return kInf;
  }

  alphas[0] = 0;
  for (int t = 1; t < T; t++) {
    alphas[idx2(t, 0, maxU)] = alphas[idx2(t-1, 0, maxU)]
      + emissions[idx2(t-1, blank, V)]
      + predictions[idx2(0, blank, V)]
      - logNorms[idx2(t-1, 0, maxU)];
  }

  for (int u = 1; u < U; u++) {
    alphas[idx2(0, u, maxU)] = alphas[idx2(0, u-1, maxU)]
      + emissions[idx2(0, labels[u-1], V)]
      + predictions[idx2(u-1, labels[u-1], V)]
      - logNorms[idx2(0, u-1, maxU)];
  }

  for (int t = 1; t < T; t++) {
    for (int u = 1; u < U; u++) {
      int prevIdx = idx2(t-1, u, maxU);
      float noEmit = alphas[prevIdx]
        + emissions[idx2(t-1, blank, V)]
        + predictions[idx2(u, blank, V)]
        - logNorms[prevIdx];
      prevIdx = idx2(t, u-1, maxU);
      float emit = alphas[prevIdx]
        + emissions[idx2(t, labels[u-1], V)]
        + predictions[idx2(u-1, labels[u-1], V)]
        - logNorms[prevIdx];
      alphas[idx2(t, u, maxU)] = logSumExp(emit, noEmit);
    }
  }
  float cost = alphas[idx2(T-1, U-1, maxU)]
    + emissions[idx2(T-1, blank, V)]
    + predictions[idx2(U-1, blank, V)]
    - logNorms[idx2(T-1, U-1, maxU)];

  return -cost;
}

void backwardSingle(
    const float* emissions,
    const float* predictions,
    float* egrads,
    float* pgrads,
    float* dalphas,
    const float* alphas,
    const float* logNorms,
    const int* labels,
    int blank, int T,
    int U, int maxU, int V) {
  if (T == 0 || U == 0) {
    return;
  }
  dalphas[idx2(T-1, U-1, maxU)] = 1.0;
  egrads[idx2(T-1, blank, V)] = -1.0;
  pgrads[idx2(U-1, blank, V)] = -1.0;

  for (int t = T-2; t >= 0; t--) {
    float g = dalphas[idx2(t+1, U-1, maxU)] * std::exp(
        alphas[idx2(t, U-1, maxU)]
        + emissions[idx2(t, blank, V)]
        + predictions[idx2(U-1, blank, V)]
        - logNorms[idx2(t, U-1, maxU)]
        - alphas[idx2(t+1, U-1, maxU)]);
    dalphas[idx2(t, U-1, maxU)] = g;
    egrads[idx2(t, blank, V)] -= g;
    pgrads[idx2(U-1, blank, V)] -= g;
  }
  for (int u = U-2; u >= 0; u--) {
    float g = dalphas[idx2(T-1, u+1, maxU)] * std::exp(
        alphas[idx2(T-1, u, maxU)]
        + emissions[idx2(T-1, labels[u], V)]
        + predictions[idx2(u, labels[u], V)]
        - logNorms[idx2(T-1, u, maxU)]
        - alphas[idx2(T-1, u+1, maxU)]);
    dalphas[idx2(T-1, u, maxU)] = g;
    egrads[idx2(T-1, labels[u], V)] -= g;
    pgrads[idx2(u, labels[u], V)] -= g;
  }

  for (int t = T-2; t >= 0; t--) {
    for (int u = U-2; u >= 0; u--) {
      float noEmit = dalphas[idx2(t+1, u, maxU)] * std::exp(
          alphas[idx2(t, u, maxU)]
          + emissions[idx2(t, blank, V)]
          + predictions[idx2(u, blank, V)]
          - logNorms[idx2(t, u, maxU)]
          - alphas[idx2(t+1, u, maxU)]);
      float emit = dalphas[idx2(t, u+1, maxU)] * std::exp(
          alphas[idx2(t, u, maxU)]
          + emissions[idx2(t, labels[u], V)]
          + predictions[idx2(u, labels[u], V)]
          - logNorms[idx2(t, u, maxU)]
          - alphas[idx2(t, u+1, maxU)]);
      dalphas[idx2(t, u, maxU)] = noEmit + emit;
      egrads[idx2(t, blank, V)] -= noEmit;
      pgrads[idx2(u, blank, V)] -= noEmit;
      egrads[idx2(t, labels[u], V)] -= emit;
      pgrads[idx2(u, labels[u], V)] -= emit;
   }
  }
}

void viterbiSingle(
    const float* emissions,
    const float* predictions,
    int* labels,
    int blank, int T,
    int U, int V) {
  if (T == 0 || U == 0) {
    return;
  }
  int* paths = (int*) malloc(T * U * sizeof(int));
  float* scores = (float*) malloc(T * U * sizeof(float));

  auto getMax = [emissions, predictions, V, blank](int t, int u) {
    float maxScore = kNegInf;
    int maxIdx = 0;
    for (int v = 0; v < V; ++v) {
      if (v != blank) {
        float score = emissions[idx2(t, v, V)] + predictions[idx2(u, v, V)];
        if (score > maxScore) {
          maxScore = score;
          maxIdx = v;
        }
      }
    }
    return std::make_pair(maxScore, maxIdx);
  };

  scores[0] = 0;
  for (int t = 1; t < T; t++) {
    scores[idx2(t, 0, U)] = scores[idx2(t-1, 0, U)]
      + emissions[idx2(t-1, blank, V)]
      + predictions[idx2(0, blank, V)];
    paths[idx2(t, 0, U)] = blank;
  }

  for (int u = 1; u < U; u++) {
    auto maxAndIdx = getMax(0, u - 1);
    scores[idx2(0, u, U)] = maxAndIdx.first + scores[idx2(0, u-1, U)];
    paths[idx2(0, u, U)] = maxAndIdx.second;
  }

  for (int t = 1; t < T; t++) {
    for (int u = 1; u < U; u++) {
      float noEmit = scores[idx2(t-1, u, U)]
        + emissions[idx2(t-1, blank, V)]
        + predictions[idx2(u, blank, V)];
      float emit = scores[idx2(t, u-1, U)];
      auto maxAndIdx = getMax(t, u-1);
      emit += maxAndIdx.first;

      if (emit > noEmit) {
        scores[idx2(t, u, U)] = emit;
        paths[idx2(t, u, U)] = maxAndIdx.second;
      } else {
        scores[idx2(t, u, U)] = noEmit;
        paths[idx2(t, u, U)] = blank;
      }
    }
  }

  int t = T - 1;
  int u = U - 1;
  while (u > 0) {
    int l = paths[idx2(t, u, U)];
    if (l == blank) {
      t -= 1;
    } else {
      labels[(u--) - 1] = l;
    }
  }
  free(paths);
  free(scores);
}

} // namespace

namespace cpu {

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
    int blank) {
#pragma omp parallel for
  for (int mb = 0; mb < batchSize; ++mb) {
    int T = inputLengths[mb]; // Length of utterance (time)
    int U = labelLengths[mb] + 1; // Length of transcription
    int eOffset = mb * maxInputLength * alphabetSize;
    int pOffset = mb * maxLabelLength * alphabetSize;
    int labelOffset = mb * (maxLabelLength - 1);
    costs[mb] = forwardSingle(
        emissions + eOffset,
        predictions + pOffset,
        alphas + mb * maxInputLength * maxLabelLength,
        logNorms + mb * maxInputLength * maxLabelLength,
        labels + labelOffset,
        blank, T, U, maxLabelLength, alphabetSize);
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
    int blank) {
#pragma omp parallel for
  for (int mb = 0; mb < batchSize; ++mb) {
    int T = inputLengths[mb]; // Length of utterance (time)
    int U = labelLengths[mb] + 1; // Length of transcription
    int eOffset = mb * maxInputLength * alphabetSize;
    int pOffset = mb * maxLabelLength * alphabetSize;
    int labelOffset = mb * (maxLabelLength - 1);
    int lnOffset = mb * maxInputLength * maxLabelLength;
    memset(
        (void*)(egrads + eOffset),
        0,
        sizeof(float) * maxInputLength * alphabetSize);
    memset(
        (void*)(pgrads + pOffset),
        0,
        sizeof(float) * maxLabelLength * alphabetSize);
    memset(
        (void*)(lngrads + lnOffset),
        0,
        sizeof(float) * maxInputLength * maxLabelLength);
    backwardSingle(
        emissions + eOffset,
        predictions + pOffset,
        egrads + eOffset,
        pgrads + pOffset,
        lngrads + lnOffset,
        alphas + lnOffset,
        logNorms + lnOffset,
        labels + labelOffset,
        blank, T, U, maxLabelLength, alphabetSize);
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
    int blank) {
#pragma omp parallel for
  for (int mb = 0; mb < batchSize; ++mb) {
    int T = inputLengths[mb]; // Length of utterance (time)
    int U = labelLengths[mb] + 1; // Length of transcription
    int eOffset = mb * maxInputLength * alphabetSize;
    int pOffset = mb * maxLabelLength * alphabetSize;
    int labelOffset = mb * (maxLabelLength - 1);
    viterbiSingle(
        emissions + eOffset,
        predictions + pOffset,
        labels + labelOffset,
        blank, T, U, alphabetSize);
  }
}

} // namespace cpu
