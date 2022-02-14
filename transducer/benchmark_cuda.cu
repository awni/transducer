#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "transducer.h"
#include "transducer_cuda.h"

#define TIME(FUNC) \
  { \
    std::cout << "Timing " << #FUNC << " ...  " << std::flush; \
    std::cout << std::setprecision(5) << timeit(FUNC) << " msec" << std::endl; \
  }

#define milliseconds(x) \
  (std::chrono::duration_cast<std::chrono::nanoseconds>(x).count() / 1e6)
#define timeNow() std::chrono::high_resolution_clock::now()

float* deviceAlloc(size_t size) {
  float* ptr;
  CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(float) * size));
  return ptr;
}

int* deviceCopy(const std::vector<int>& hostVec) {
  int* dptr;
  CUDA_CHECK(cudaMalloc((void**)&dptr, sizeof(float) * hostVec.size()));
  CUDA_CHECK(
      cudaMemcpy((void*) dptr,
      (void*) hostVec.data(),
      sizeof(int) * hostVec.size(),
      cudaMemcpyHostToDevice));
  return dptr;
}

float* deviceCopy(const std::vector<float>& hostVec) {
  float* dptr = deviceAlloc(hostVec.size());
  CUDA_CHECK(
      cudaMemcpy((void*) dptr,
      (void*) hostVec.data(),
      sizeof(float) * hostVec.size(),
      cudaMemcpyHostToDevice));
  return dptr;
}

void deviceFree(float* dptr) {
  CUDA_CHECK(cudaFree((void*)dptr));
}

void deviceFree(int* dptr) {
  CUDA_CHECK(cudaFree((void*)dptr));
}

double timeit(std::function<void()> fn) {
  // warmup
  for (int i = 0; i < 5; ++i) {
    fn();
    cudaDeviceSynchronize();
  }

  int numIters = 100;
  auto start = timeNow();
  for (int i = 0; i < numIters; i++) {
    fn();
    cudaDeviceSynchronize();
  }
  auto end = timeNow();
  return milliseconds(end - start) / static_cast<double>(numIters);
}

void timeTransducer(int B, int T, int U, int V) {
  std::vector<float> emissions(B * T * V);
  std::generate(emissions.begin(), emissions.end(), std::rand);

  std::vector<float> predictions(B * (U + 1) * V);
  std::generate(predictions.begin(), predictions.end(), std::rand);

  std::vector<int> labels(B * U);
  std::generate(
      labels.begin(),
      labels.end(),
      [V](){ return std::rand() % V; });

  std::vector<int> inputLengths(B, T);
  std::vector<int> labelLengths(B, U);

  auto emissionsD = deviceCopy(emissions);
  auto predictionsD = deviceCopy(predictions);
  auto labelsD = deviceCopy(labels);
  auto inputLengthsD = deviceCopy(inputLengths);
  auto labelLengthsD = deviceCopy(labelLengths);
  auto costs = deviceAlloc(B);
  auto alphas = deviceAlloc(B * T * (U + 1));
  auto logNorms = deviceAlloc(B * T * (U + 1));

  auto transducerForward = [=]() {
      forward(
          emissionsD,
          predictionsD,
          costs,
          alphas,
          logNorms,
          labelsD,
          inputLengthsD,
          labelLengthsD,
          B, T, U + 1, V, 0, true);
  };
  TIME(transducerForward);

  auto egrads = deviceAlloc(B * T * V);
  auto pgrads = deviceAlloc(B * U * V);
  auto transducerBackward = [=]() {
      backward(
          emissionsD,
          predictionsD,
          egrads,
          pgrads,
          alphas,
          logNorms,
          labelsD,
          inputLengthsD,
          labelLengthsD,
          B, T, U + 1, V, 0, true);
  };
  TIME(transducerBackward);

  deviceFree(emissionsD);
  deviceFree(predictionsD);
  deviceFree(labelsD);
  deviceFree(inputLengthsD);
  deviceFree(labelLengthsD);
  deviceFree(costs);
  deviceFree(alphas);
  deviceFree(logNorms);
  deviceFree(egrads);
  deviceFree(pgrads);
}

int main() {
  int B = 16;
  int T = 150;
  int U = 40;
  int V = 5000;
  timeTransducer(B, T, U, V);
}
