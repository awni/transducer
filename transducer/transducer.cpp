
#if defined(OPENMP)
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>

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

int cumsum(int *lens, int num) {
  int sum = 0;
  for (int i = 0; i < num; i++) {
    sum += lens[i];
  }
  return sum;
}

float* logNormScores(float* emissions, float* predictions, int T, int U, int V) {
  float* logNorms = (float*) malloc(T * U * sizeof(float));
  for (int t = 0; t < T; ++t) {
    for (int u = 0; u < U; ++u) {
      float maxScore = kNegInf;
      for (int v = 0; v < V; ++v) {
        maxScore = std::max(maxScore, emissions[idx2(t, v, V)] + predictions[idx2(u, v, V)]);
      }
      float expSum = 0.0;
      for (int v = 0; v < V; ++v) {
        expSum += std::exp(emissions[idx2(t, v, V)] + predictions[idx2(u, v, V)] - maxScore);
      }
      logNorms[idx2(t, u, U)] = std::log(expSum) + maxScore;
    }
  }
  return logNorms;
}

float costAndGradSingle(
    float* emissions,
    float* predictions,
    float* egrads,
    float* pgrads,
    int* labels,
    int blank, int T,
    int U, int V) {
    
  auto logNorms = logNormScores(emissions, predictions, T, U, V); 

  // Forward pass
  float* alphas = (float*) malloc(T * U * sizeof(float));
  alphas[0] = 0;
  for (int t = 1; t < T; t++) {
    alphas[idx2(t, 0, U)] = alphas[idx2(t-1, 0, U)]
      + emissions[idx2(t-1, blank, V)]
      + predictions[idx2(0, blank, V)] 
      - logNorms[idx2(t-1, 0, U)];
  }

  for (int u = 1; u < U; u++) {
    alphas[idx2(0, u, U)] = alphas[idx2(0, u-1, U)] 
      + emissions[idx2(0, labels[u-1], V)]
      + predictions[idx2(u-1, labels[u-1], V)] 
      - logNorms[idx2(0, u-1, U)];
  }

  for (int t = 1; t < T; t++) {
    for (int u = 1; u < U; u++) {
      int prevIdx = idx2(t-1, u, U);
      float noEmit = alphas[prevIdx]
        + emissions[idx2(t-1, blank, V)]
        + predictions[idx2(u, blank, V)] 
        - logNorms[prevIdx];
      prevIdx = idx2(t, u-1, U);
      float emit = alphas[prevIdx]
        + emissions[idx2(t, labels[u-1], V)]
        + predictions[idx2(u-1, labels[u-1], V)]
        - logNorms[prevIdx];
      alphas[idx2(t, u, U)] = logSumExp(emit, noEmit);
    }
  }
  float cost = alphas[idx2(T-1, U-1, U)]
    + emissions[idx2(T-1, blank, V)]
    + predictions[idx2(U-1, blank, V)]
    - logNorms[idx2(T-1, U-1, U)];

  // Backward pass
  float* dalphas = (float*) malloc(T * U * sizeof(float));
  dalphas[idx2(T-1, U-1, U)] = -1.0;
  egrads[idx2(T-1, blank, V)] = -1.0;
  pgrads[idx2(U-1, blank, V)] = -1.0;

  for (int t = T-2; t >= 0; t--) {
    float g = dalphas[idx2(t+1, U-1, U)] * std::exp(
        alphas[idx2(t, U-1, U)]
        + emissions[idx2(t, blank, V)]
        + predictions[idx2(U-1, blank, V)] 
        - logNorms[idx2(t, U-1, U)]
        - alphas[idx2(t+1, U-1, U)]);
    dalphas[idx2(t, U-1, U)] = g;
    egrads[idx2(t, blank, V)] += g;
    pgrads[idx2(U-1, blank, V)] += g;
  }
  for (int u = U-2; u >= 0; u--) {
    float g = dalphas[idx2(T-1, u+1, U)] * std::exp(
        alphas[idx2(T-1, u, U)]
        + emissions[idx2(T-1, labels[u], V)]
        + predictions[idx2(u, labels[u], V)] 
        - logNorms[idx2(T-1, u, U)]
        - alphas[idx2(T-1, u+1, U)]);
    dalphas[idx2(T-1, u, U)] = g;
    egrads[idx2(T-1, labels[u], V)] += g;
    pgrads[idx2(u, labels[u], V)] += g;
  }

  for (int t = T-2; t >= 0; t--) {
    for (int u = U-2; u >= 0; u--) {
      float noEmit = dalphas[idx2(t+1, u, U)] * std::exp(
          alphas[idx2(t, u, U)]
          + emissions[idx2(t, blank, V)]
          + predictions[idx2(u, blank, V)] 
          - logNorms[idx2(t, u, U)]
          - alphas[idx2(t+1, u, U)]);
      float emit = dalphas[idx2(t, u+1, U)] * std::exp(
          alphas[idx2(t, u, U)]
          + emissions[idx2(t, labels[u], V)]
          + predictions[idx2(u, labels[u], V)] 
          - logNorms[idx2(t, u, U)]
          - alphas[idx2(t, u+1, U)]);
      dalphas[idx2(t, u, U)] = noEmit + emit;
      egrads[idx2(t, blank, V)] += noEmit;
      pgrads[idx2(u, blank, V)] += noEmit;
      egrads[idx2(t, labels[u], V)] += emit;
      pgrads[idx2(u, labels[u], V)] += emit;
   }
  }

  // Accumulate gradients
  for (int t = 0; t < T; ++t) {
    for (int u = 0; u < U; ++u) {
      float dalpha = dalphas[idx2(t, u, U)];
      float logNorm = logNorms[idx2(t, u, U)];
      for (int v = 0; v < V; ++v) {
        float score = std::exp(
            emissions[idx2(t, v, V)] + predictions[idx2(u, v, V)] - logNorm);
        egrads[idx2(t, v, V)] -= dalpha * score; 
        pgrads[idx2(u, v, V)] -= dalpha * score; 
      }
    }
  }

  // Cleanup
  free(alphas);
  free(dalphas);
  free(logNorms);

  return -cost;
}
} // namespace

void costAndGrad(
    float* emissions,
    float* predictions,
    float* egrads,
    float* pgrads,
    float* costs,
    int* labels,
    int* inputLengths,
    int* labelLengths,
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
    int labelOffset = cumsum(labelLengths, mb);
    costs[mb] = costAndGradSingle(
        emissions + eOffset,
        predictions + pOffset,
        egrads + eOffset,
        pgrads + pOffset,
        labels + labelOffset,
        blank, T, U, alphabetSize);
  }
}
