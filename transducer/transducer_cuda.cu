#include <limits>
#include <stdexcept>
#include "transducer_cuda.h"

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
      int prevIdx = idx2(t-1, u, maxU);
      float noEmit = (t == 0) ? kNegInf : 
          alphas[prevIdx] +
          emissions[idx2(t-1, blank, V)] +
          predictions[idx2(u, blank, V)] -
          logNorms[prevIdx];
      prevIdx = idx2(t, u-1, maxU);
      float emit = (u == 0) ? kNegInf :
          alphas[prevIdx] +
          emissions[idx2(t, labels[u-1], V)] +
          predictions[idx2(u-1, labels[u-1], V)] -
          logNorms[prevIdx];
      alphas[idx2(t, u, maxU)] = logSumExp(emit, noEmit);
    }
    __syncthreads();
  }
  if (tidx == 0) {
    costs[mb] = -(alphas[idx2(T-1, U-1, maxU)]
      + emissions[idx2(T-1, blank, V)]
      + predictions[idx2(U-1, blank, V)]
      - logNorms[idx2(T-1, U-1, maxU)]);
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
    dalphas[idx2(T-1, U-1, maxU)] = 1.0f;
    egrads[idx2(T-1, blank, V)] = -1.0f;
    pgrads[idx2(U-1, blank, V)] = -1.0f;
  }
  __syncthreads();

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
      float alpha_ln = alphas[idx2(t, u, maxU)] - logNorms[idx2(t, u, maxU)];
      float noEmit = 0.0f;
      if (t < (T - 1)) {
        noEmit = dalphas[idx2(t+1, u, maxU)] *
          expf(alpha_ln +
              emissions[idx2(t, blank, V)] +
              predictions[idx2(u, blank, V)] -
              alphas[idx2(t+1, u, maxU)]);
        egrads[idx2(t, blank, V)] -= noEmit;
        pgrads[idx2(u, blank, V)] -= noEmit;
      }
      float emit = 0.0f;
      if (u < (U - 1)) {
        emit = dalphas[idx2(t, u+1, maxU)] *
          expf(alpha_ln +
              emissions[idx2(t, labels[u], V)] +
              predictions[idx2(u, labels[u], V)] -
              alphas[idx2(t, u+1, maxU)]);
        egrads[idx2(t, labels[u], V)] -= emit;
        pgrads[idx2(u, labels[u], V)] -= emit;
      }
      dalphas[idx2(t, u, maxU)] = noEmit + emit;
    }
    __syncthreads();
  }
}

__global__
void viterbiKernel(
    const float* emissions,
    const float* predictions,
    const float* logNorms,
    float* scores,
    int* paths,
    int* labels,
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
  emissions += maxT * V * mb;
  predictions += maxU * V * mb;
  logNorms += maxT * maxU * mb;
  scores += maxT * maxU * mb;
  paths += maxT * maxU * mb;
  labels += (maxU - 1) * mb;

  if (tidx == 0) {
    scores[0] = 0.0f;
  }
  __syncthreads();

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
      float noEmit = (t == 0) ? kNegInf : 
          scores[idx2(t-1, u, U)] +
          emissions[idx2(t-1, blank, V)] +
          predictions[idx2(u, blank, V)] -
          logNorms[idx2(t-1, u, maxU)];
      float emit;
      float maxIdx = 0;
      if (u == 0) {
        emit = kNegInf;
      } else {
        emit = scores[idx2(t, u-1, U)] - logNorms[idx2(t, u-1, maxU)];
        float maxScore = kNegInf;
        for (int v = 0; v < V; ++v) {
          if (v == blank) {
            continue;
          }
          float score = emissions[idx2(t, v, V)] + predictions[idx2(u-1, v, V)];
          if (score > maxScore) {
            maxScore = score;
            maxIdx = v;
          }
        }
        emit += maxScore;
      }
      if (emit > noEmit) {
        scores[idx2(t, u, U)] = emit;
        paths[idx2(t, u, U)] = maxIdx;
      } else {
        scores[idx2(t, u, U)] = noEmit;
        paths[idx2(t, u, U)] = blank;
      }
    }
    __syncthreads();
  }
  if (tidx == 0) {
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
  CUDA_CHECK(cudaMemsetAsync(
      (void*)egrads, 0, sizeof(float) * batchSize * maxInputLength * alphabetSize, 0));
  CUDA_CHECK(cudaMemsetAsync(
      (void*)pgrads, 0, sizeof(float) * batchSize * maxLabelLength * alphabetSize, 0));
  CUDA_CHECK(cudaMemsetAsync(
      (void*)lngrads, 0, sizeof(float) * batchSize * maxInputLength * maxLabelLength, 0));
  int threads = std::min(1024, divUp(std::min(maxInputLength, maxLabelLength), 32) * 32);
  backwardKernel<<<batchSize, threads>>>(
    emissions,
    predictions,
    egrads,
    pgrads,
    alphas,
    lngrads,
    logNorms,
    labels,
    inputLengths,
    labelLengths,
    maxInputLength,
    maxLabelLength,
    alphabetSize,
    blank);
}

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
    int blank) {
  float* scores;
  int* paths;
  CUDA_CHECK(cudaMallocAsync(
      (void**)&scores,
      sizeof(float) * batchSize * maxInputLength * maxLabelLength, 0));
  CUDA_CHECK(cudaMallocAsync(
      (void**)&paths,
      sizeof(int) * batchSize * maxInputLength * maxLabelLength, 0));
  {
    int threads = std::min(1024, divUp(std::min(maxInputLength, maxLabelLength), 32) * 32);
    viterbiKernel<<<batchSize, threads>>>(
      emissions,
      predictions,
      logNorms,
      scores,
      paths,
      labels,
      inputLengths,
      labelLengths,
      maxInputLength,
      maxLabelLength,
      alphabetSize,
      blank);
  }
  CUDA_CHECK(cudaFreeAsync(scores, 0));
  CUDA_CHECK(cudaFreeAsync(paths, 0));
}

} // namespace cuda
