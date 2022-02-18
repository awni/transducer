#include <limits>
#include <stdexcept>
#include "transducer_cuda.h"

#define NT 8
#define ELEMS_PER_THREAD 4

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
  __shared__ float eTile[NT * ELEMS_PER_THREAD][NT],
                   pTile[NT][NT * ELEMS_PER_THREAD];
  int ts = blockIdx.x * blockDim.x * ELEMS_PER_THREAD;
  int us = blockIdx.y * blockDim.y * ELEMS_PER_THREAD;
  int mb = blockIdx.z;
  int T = inputLengths[mb];
  int U = labelLengths[mb] + 1;
  if (ts >= T || us >= U) {
    return;
  }
  emissions += maxInputLength * alphabetSize * mb;
  predictions += maxLabelLength * alphabetSize * mb;
  logNorms += maxInputLength * maxLabelLength * mb;

  float maxScores[ELEMS_PER_THREAD][ELEMS_PER_THREAD];
  float scores[ELEMS_PER_THREAD][ELEMS_PER_THREAD];
  for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
    for (int j = 0; j < ELEMS_PER_THREAD; ++j) {
      maxScores[i][j] = kNegInf;
      scores[i][j] = 0.0;
    }
  }
  for (int i = 0; i < alphabetSize; i += NT) {
    // Load tiles into shared memory
    for (int j = 0; j < ELEMS_PER_THREAD; j++) {
      int tidy = j * blockDim.y + threadIdx.y;
      if ((ts + threadIdx.y) < T && (i + threadIdx.x) < alphabetSize) {
        eTile[tidy][threadIdx.x] = emissions[(ts + tidy) * alphabetSize + i + threadIdx.x];
      } else {
        eTile[tidy][threadIdx.x] = kNegInf;
      }
      if ((us + tidy) < U && (i + threadIdx.x) < alphabetSize) {
        pTile[threadIdx.x][tidy] = predictions[(us + tidy) * alphabetSize + i + threadIdx.x];
      } else {
        pTile[threadIdx.x][tidy] = kNegInf;
      }
    }
    __syncthreads();

    // Process tiles
    for (int j = 0; j < ELEMS_PER_THREAD; j++) {
      int tidy = j * blockDim.y + threadIdx.y;
      for (int k = 0; k < ELEMS_PER_THREAD; k++) {
        int tidx = k * blockDim.x + threadIdx.x;
        for (int l = 0; l < NT; ++l) {
          maxScores[j][k] = max(maxScores[j][k], eTile[tidy][l] + pTile[l][tidx]);
        }
      }
    }
    __syncthreads();
  }

  for (int i = 0; i < alphabetSize; i += NT) {
    // Load tiles into shared memory
    for (int j = 0; j < ELEMS_PER_THREAD; j++) {
      int tidy = j * blockDim.y + threadIdx.y;
      if ((ts + threadIdx.y) < T && (i + threadIdx.x) < alphabetSize) {
        eTile[tidy][threadIdx.x] = emissions[(ts + tidy) * alphabetSize + i + threadIdx.x];
      } else {
        eTile[tidy][threadIdx.x] = kNegInf;
      }
      if ((us + tidy) < U && (i + threadIdx.x) < alphabetSize) {
        pTile[threadIdx.x][tidy] = predictions[(us + tidy) * alphabetSize + i + threadIdx.x];
      } else {
        pTile[threadIdx.x][tidy] = kNegInf;
      }
    }
    __syncthreads();

    // Process tiles
    for (int j = 0; j < ELEMS_PER_THREAD; j++) {
      int tidy = j * blockDim.y + threadIdx.y;
      for (int k = 0; k < ELEMS_PER_THREAD; k++) {
        int tidx = k * blockDim.x + threadIdx.x;
        for (int l = 0; l < NT; ++l) {
          scores[j][k] += __expf(eTile[tidy][l] + pTile[l][tidx] - maxScores[j][k]);
        }
      }
    }
    __syncthreads();
  }

  for (int j = 0; j < ELEMS_PER_THREAD; j++) {
    int t = ts + j * blockDim.y + threadIdx.y;
    for (int k = 0; k < ELEMS_PER_THREAD; k++) {
      int u = us + k * blockDim.x + threadIdx.x;
      if (t < T && u < U) {
        if (maxScores[j][k] == kInf || maxScores[j][k] == kNegInf) {
          scores[j][k] = maxScores[j][k];
        } else {
          scores[j][k] = logf(scores[j][k]) + maxScores[j][k];
        }
        logNorms[idx2(t, u, U)] = scores[j][k];
      }
    }
  }
}

__global__
void accumulateGradsKernel(
      const float* A,
      const float* B,
      float* dA,
      const float* dalphas,
      const float* logNorms,
      const int* lensA,
      const int* lensB,
      int maxLenA,
      int maxLenB,
      int alphabetSize,
      bool transpose) {
  __shared__ float bTile[NT][NT * ELEMS_PER_THREAD],
                   lTile[NT * ELEMS_PER_THREAD][NT],
                   dTile[NT * ELEMS_PER_THREAD][NT];
  int ts = blockIdx.x * blockDim.x * ELEMS_PER_THREAD;
  int vs = blockIdx.y * blockDim.y * ELEMS_PER_THREAD;
  int mb = blockIdx.z;
  int T = lensA[mb];
  int U = lensB[mb];
  if (transpose) {
    T += 1;
  } else {
    U += 1;
  }
  if (ts >= T || vs >= alphabetSize) {
    return;
  }
  int offset = maxLenA * alphabetSize * mb;
  A += offset;
  dA += offset;
  offset = maxLenB * alphabetSize * mb;
  B += offset;
  offset = maxLenA * maxLenB * mb;
  dalphas += offset;
  logNorms += offset;

  float ascores[ELEMS_PER_THREAD][ELEMS_PER_THREAD];
  float grads[ELEMS_PER_THREAD][ELEMS_PER_THREAD];
  for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
    int t = ts + i * blockDim.y + threadIdx.y;
    for (int j = 0; j < ELEMS_PER_THREAD; ++j) {
      int v = vs + j * blockDim.x + threadIdx.x;
      if (v < alphabetSize && t < T) {
        ascores[i][j] = A[idx2(t, v, alphabetSize)];
        grads[i][j] = 0.0f;
      }
    }
  }
  for (int i = 0; i < U; i += NT) {
    // Load tiles into shared memory
    for (int j = 0; j < ELEMS_PER_THREAD; j++) {
      int tidx = j * blockDim.x + threadIdx.x;
      if ((vs + tidx) < alphabetSize && (i + threadIdx.y) < U) {
        bTile[threadIdx.y][tidx] = B[(i + threadIdx.y) * alphabetSize + vs + tidx];
      } else {
        bTile[threadIdx.y][tidx] = 0.0f;
      }

      if (transpose) {
        int tidy = i + threadIdx.y;
        int tidx = j * blockDim.x + threadIdx.x;
        if ((ts + tidx) < T && tidy < U) {
          lTile[tidx][threadIdx.y] = logNorms[tidy * T + ts + tidx];
          dTile[tidx][threadIdx.y] = dalphas[tidy * T + ts + tidx];
        } else {
          lTile[tidx][threadIdx.y] = 0.0f;
          dTile[tidx][threadIdx.y]  = 0.0f;
        }
      } else {
        int tidx = i + threadIdx.x;
        int tidy = j * blockDim.y + threadIdx.y;
        if ((ts + tidy) < T && tidx < U) {
          lTile[tidy][threadIdx.x] = logNorms[(ts + tidy) * U + tidx];
          dTile[tidy][threadIdx.x] = dalphas[(ts + tidy) * U + tidx];
        } else {
          lTile[tidy][threadIdx.x] = 0.0f;
          dTile[tidy][threadIdx.x] = 0.0f;
        }
      }
    }
    __syncthreads();

    // Process tiles
    for (int j = 0; j < ELEMS_PER_THREAD; j++) {
      int tidy = j * blockDim.y + threadIdx.y;
      for (int k = 0; k < ELEMS_PER_THREAD; ++k) {
        int tidx = k * blockDim.x + threadIdx.x;
        for (int l = 0; l < NT; ++l) {
          grads[j][k] += dTile[tidy][l] * expf(ascores[j][k] - lTile[tidy][l] + bTile[l][tidx]);
        }
      }
    }
    __syncthreads();
  }

  for (int j = 0; j < ELEMS_PER_THREAD; j++) {
    int t = ts + j * blockDim.y + threadIdx.y;
    for (int k = 0; k < ELEMS_PER_THREAD; k++) {
      int v = vs + k * blockDim.x + threadIdx.x;
      if (t < T && v < alphabetSize) {
        dA[idx2(t, v, alphabetSize)] -= grads[j][k];
      }
    }
  }
}

__global__
void forwardKernel(
    const float* emissions,
    const float* predictions,
    float* costs,
    float* alphas,
    const float* logNorms,
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
  if (T == 0 || U == 0) {
    costs[mb] = kInf;
    return;
  }
  emissions += maxT * V * mb;
  predictions += maxU * V * mb;
  logNorms += maxT * maxU * mb;
  alphas += maxT * maxU * mb;
  labels += (maxU - 1) * mb;

  if (tidx == 0) {
    alphas[0] = 0.0f;
  }
  __syncthreads();

  // Compute label offset
  for (int i = 1; i < (T + U - 1); ++i) {
    int t, u;
    if (i >= T) {
      t = T - 1 - tidx;
      u = (i - T + 1) + tidx;
    } else {
      t = i - tidx;
      u = tidx;
    }
    for (; t >= 0 && u < U; t -= blockDim.x, u += blockDim.x) {
      int prevIdx = idx2(t-1, u, U);
      float noEmit = (t == 0) ? kNegInf : 
          alphas[prevIdx] +
          emissions[idx2(t-1, blank, V)] +
          predictions[idx2(u, blank, V)] -
          logNorms[prevIdx];
      prevIdx = idx2(t, u-1, U);
      float emit = (u == 0) ? kNegInf :
          alphas[prevIdx] +
          emissions[idx2(t, labels[u-1], V)] +
          predictions[idx2(u-1, labels[u-1], V)] -
          logNorms[prevIdx];
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

__global__
void backwardKernel(
    const float* emissions,
    const float* predictions,
    float* egrads,
    float* pgrads,
    const float* alphas,
    float* dalphas,
    const float* logNorms,
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
  if (T == 0 || U == 0) {
    return;
  }
  int offset = maxT * V * mb;
  emissions += offset;
  egrads += offset;
  offset = maxU * V * mb;
  predictions += offset;
  pgrads += offset;
  offset = maxT * maxU * mb;
  logNorms += offset;
  alphas += offset;
  dalphas += offset;
  labels += (maxU - 1) * mb;

  if (tidx == 0) {
    dalphas[idx2(T-1, U-1, U)] = -1.0f;
    egrads[idx2(T-1, blank, V)] = -1.0f;
    pgrads[idx2(U-1, blank, V)] = -1.0f;
  }
  __syncthreads();

  // Compute label offset
  for (int i = (T + U - 3); i >= 0; --i) {
    int t, u;
    if (i >= T) {
      t = T - 1 - tidx;
      u = (i - T + 1) + tidx;
    } else {
      t = i - tidx;
      u = tidx;
    }
    for (; t >= 0 && u < U; t -= blockDim.x, u += blockDim.x) {
      float alpha_ln = alphas[idx2(t, u, U)] - logNorms[idx2(t, u, U)];
      float noEmit = 0.0f;
      if (t < (T - 1)) {
        noEmit = dalphas[idx2(t+1, u, U)] *
          expf(alpha_ln +
              emissions[idx2(t, blank, V)] +
              predictions[idx2(u, blank, V)] -
              alphas[idx2(t+1, u, U)]);
        egrads[idx2(t, blank, V)] += noEmit;
        pgrads[idx2(u, blank, V)] += noEmit;
      }
      float emit = 0.0f;
      if (u < (U - 1)) {
        emit = dalphas[idx2(t, u+1, U)] *
          expf(alpha_ln +
              emissions[idx2(t, labels[u], V)] +
              predictions[idx2(u, labels[u], V)] -
              alphas[idx2(t, u+1, U)]);
        egrads[idx2(t, labels[u], V)] += emit;
        pgrads[idx2(u, labels[u], V)] += emit;
      }
      dalphas[idx2(t, u, U)] = noEmit + emit;
    }
    __syncthreads();
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
  dim3 blocks(
      divUp(maxInputLength, ELEMS_PER_THREAD * NT),
      divUp(maxLabelLength, ELEMS_PER_THREAD * NT),
      batchSize);
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
  int threads = std::min(1024, divUp(std::min(maxInputLength, maxLabelLength), 32) * 32);
  forwardKernel<<<batchSize, threads>>>(
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
  float* dalphas;
  CUDA_CHECK(cudaMallocAsync(
      (void**)&dalphas,
      sizeof(float) * batchSize * maxInputLength * maxLabelLength, 0));
  CUDA_CHECK(cudaMemsetAsync(
      (void*)egrads, 0, sizeof(float) * batchSize * maxInputLength * alphabetSize, 0));
  CUDA_CHECK(cudaMemsetAsync(
      (void*)pgrads, 0, sizeof(float) * batchSize * maxLabelLength * alphabetSize, 0));
  {
    int threads = std::min(1024, divUp(std::min(maxInputLength, maxLabelLength), 32) * 32);
    backwardKernel<<<batchSize, threads>>>(
      emissions,
      predictions,
      egrads,
      pgrads,
      alphas,
      dalphas,
      logNorms,
      labels,
      inputLengths,
      labelLengths,
      maxInputLength,
      maxLabelLength,
      alphabetSize,
      blank);
  }
  {
    dim3 blocks(
        divUp(maxInputLength, ELEMS_PER_THREAD * NT),
        divUp(alphabetSize, ELEMS_PER_THREAD * NT),
        batchSize);
    dim3 threads(NT, NT);
    accumulateGradsKernel<<<blocks, threads>>>(
      emissions,
      predictions,
      egrads,
      dalphas,
      logNorms,
      inputLengths,
      labelLengths,
      maxInputLength,
      maxLabelLength,
      alphabetSize,
      false);
  }
  {
    dim3 blocks(
        divUp(maxLabelLength, ELEMS_PER_THREAD * NT),
        divUp(alphabetSize, ELEMS_PER_THREAD * NT),
        batchSize);
    dim3 threads(NT, NT);
    accumulateGradsKernel<<<blocks, threads>>>(
      predictions,
      emissions,
      pgrads,
      dalphas,
      logNorms,
      labelLengths,
      inputLengths,
      maxLabelLength,
      maxInputLength,
      alphabetSize,
      true);
  }
  CUDA_CHECK(cudaFreeAsync(dalphas, 0));
}

} // namespace cuda
