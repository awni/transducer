#include <limits>
#include "transducer_cuda.h"

#define TILE_DIM 32

namespace cuda {

namespace {

__constant__ constexpr float kInf = std::numeric_limits<float>::infinity();
__constant__ constexpr float kNegInf = -std::numeric_limits<float>::infinity();

inline int divUp(int x, int y) {
  return (x + y - 1) / y;
}

__device__
inline int idx2(int x, int y, int stride) {
  return x * stride + y;
}

__device__
inline float logSumExp(float a, float b) {
  if (a == kNegInf) {
    return b;
  }
  if (b == kNegInf) {
    return a;
  }
  return fmaxf(a, b) + log1pf(expf(-fabsf(a - b)));
}

__global__
void logNormsKernel(
    const float* emissions,
    const float* predictions,
    float* logNorms,
    const int* inputLengths,
    const int* labelLengths,
    int maxInputLength,
    int maxLabelLength,
    int alphabetSize)  {
  __shared__ float eTile[TILE_DIM][TILE_DIM],
                   pTile[TILE_DIM][TILE_DIM];
  int ts = blockIdx.x * blockDim.x;
  int us = blockIdx.y * blockDim.y;
  int mb = blockIdx.z;
  int T = inputLengths[mb];
  int U = labelLengths[mb] + 1;
  if (ts >= T || us >= U) {
    return;
  }
  emissions += maxInputLength * alphabetSize * mb;
  predictions += maxLabelLength * alphabetSize * mb;
  logNorms += maxInputLength * maxLabelLength * mb;

  float maxScore = kNegInf;
  for (int i = 0; i < alphabetSize; i += TILE_DIM) {
    // Load tiles into shared memory
    if ((ts + threadIdx.y) < T && (i + threadIdx.x) < alphabetSize) {
      eTile[threadIdx.y][threadIdx.x] = emissions[(ts + threadIdx.y) * alphabetSize + i + threadIdx.x];
    } else {
      eTile[threadIdx.y][threadIdx.x] = kNegInf;
    }
    if ((us + threadIdx.y) < U && (i + threadIdx.x) < alphabetSize) {
      pTile[threadIdx.y][threadIdx.x] = predictions[(us + threadIdx.y) * alphabetSize + i + threadIdx.x];
    } else {
      pTile[threadIdx.y][threadIdx.x] = kNegInf;
    }
    __syncthreads();

    // Process tiles
    for (int j = 0; j < TILE_DIM; ++j) {
      maxScore = max(maxScore, eTile[threadIdx.x][j] + pTile[threadIdx.y][j]);
    }
    __syncthreads();
  }

  float score = 0.0;
  for (int i = 0; i < alphabetSize; i += TILE_DIM) {
    // Load tiles into shared memory
    if ((ts + threadIdx.y) < T && (i + threadIdx.x) < alphabetSize) {
      eTile[threadIdx.y][threadIdx.x] = emissions[(ts + threadIdx.y) * alphabetSize + i + threadIdx.x];
    } else {
      eTile[threadIdx.y][threadIdx.x] = kNegInf;
    }
    if ((us + threadIdx.y) < U && (i + threadIdx.x) < alphabetSize) {
      pTile[threadIdx.y][threadIdx.x] = predictions[(us + threadIdx.y) * alphabetSize + i + threadIdx.x];
    } else {
      pTile[threadIdx.y][threadIdx.x] = kNegInf;
    }
    __syncthreads();

    for (int j = 0; j < TILE_DIM; j++) {
      score += expf(eTile[threadIdx.x][j] + pTile[threadIdx.y][j] - maxScore);
    }
    __syncthreads();
  }
  if (ts + threadIdx.x < T && us + threadIdx.y < U) {
    if (maxScore == kInf || maxScore == kNegInf) {
      score = maxScore;
    } else {
      score = logf(score) + maxScore;
    }
    logNorms[idx2(ts + threadIdx.x, us + threadIdx.y, U)] = score;
  }
}

__global__
void forwardKernel(
    const float* emissions,
    const float* predictions,
    float* costs,
    float* alphas,
    float* logNorms,
    const int* labels,
    const int* inputLengths,
    const int* labelLengths,
    int maxT,
    int maxU,
    int V,
    int blank) {
  int tidx = threadIdx.x;
  int mb = blockIdx.x; 
  int T = inputLengths[mb];
  int U = labelLengths[mb] + 1;
  emissions += maxT * V * mb;
  predictions += maxU * V * mb;
  logNorms += maxT * maxU * mb;
  alphas += maxT * maxU * mb;

  __shared__ int labelOffset; 
  if (tidx == 0) {
    labelOffset = 0;
    alphas[0] = 0.0f;
  }
  __syncthreads();

  // Compute label offset for the batch
  for (int i = tidx; i < mb; i+= blockDim.x) {
    atomicAdd(&labelOffset, labelLengths[i]);
  }
  __syncthreads();
  labels += labelOffset;

  // Compute label offset
  for (int i = 1; i < T; ++i) {
    for (int t = i - tidx, u = tidx; t >= 0 && u < U; t -= blockDim.x, u += blockDim.x) {
      int prevIdx = idx2(t-1, u, U);
      float noEmit = (t == 0) ? kNegInf : 
          alphas[prevIdx]
            + emissions[idx2(t-1, blank, V)]
            + predictions[idx2(u, blank, V)]
            - logNorms[prevIdx];
      prevIdx = idx2(t, u-1, U);
      float emit = (u == 0) ? kNegInf :
        alphas[prevIdx]
          + emissions[idx2(t, labels[u-1], V)]
          + predictions[idx2(u-1, labels[u-1], V)]
          - logNorms[prevIdx];
      alphas[idx2(t, u, U)] = logSumExp(emit, noEmit);
    }
    __syncthreads();
  }
  for (int i = 1; i < U; ++i) {
    for (int t = T - 1 - tidx, u = i + tidx; t >= 0 && u < U; t -= blockDim.x, u += blockDim.x) {
      int prevIdx = idx2(t-1, u, U);
      float noEmit = (t == 0) ? kNegInf : 
          alphas[prevIdx]
            + emissions[idx2(t-1, blank, V)]
            + predictions[idx2(u, blank, V)]
            - logNorms[prevIdx];
      prevIdx = idx2(t, u-1, U);
      float emit = (u == 0) ? kNegInf :
        alphas[prevIdx]
          + emissions[idx2(t, labels[u-1], V)]
          + predictions[idx2(u-1, labels[u-1], V)]
          - logNorms[prevIdx];
      alphas[idx2(t, u, U)] = logSumExp(emit, noEmit);
    }
    __syncthreads();
  }
  if (tidx == 0) {
    costs[mb] = -(alphas[idx2(T-1, U-1, U)]
      + emissions[idx2(T-1, blank, V)]
      + predictions[idx2(U-1, blank, V)]
      - logNorms[idx2(T-1, U-1, U)]);
  }
}

} // namespace

void cudaCheck(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::ostringstream ess;
    ess << '[' << file << ':' << line
        << "] CUDA error: " << cudaGetErrorString(err);
    throw std::runtime_error(ess.str());
  }
}

void computeLogNorms(
    const float* emissions,
    const float* predictions,
    float* logNorms,
    const int* inputLengths,
    const int* labelLengths,
    int batchSize,
    int maxInputLength,
    int maxLabelLength,
    int alphabetSize) {
  const int NT = 32;
  dim3 blocks(divUp(maxInputLength, NT), divUp(maxLabelLength, NT), batchSize);
  dim3 threads(NT, NT);
  logNormsKernel<<<blocks, threads>>>(
      emissions,
      predictions,
      logNorms,
      inputLengths,
      labelLengths,
      maxInputLength,
      maxLabelLength,
      alphabetSize);
}

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
    int blank) {
  computeLogNorms(
      emissions,
      predictions,
      logNorms,
      inputLengths,
      labelLengths,
      batchSize,
      maxInputLength,
      maxLabelLength,
      alphabetSize);
  int NT = divUp(std::min(maxInputLength, maxLabelLength), 32) * 32;
  forwardKernel<<<batchSize, NT>>>(
    emissions,
    predictions,
    costs,
    alphas,
    logNorms,
    labels,
    inputLengths,
    labelLengths,
    maxInputLength,
    maxLabelLength,
    alphabetSize,
    blank);
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
    int blank) {
}

} // namespace cuda
