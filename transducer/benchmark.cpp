#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include "transducer.h"

#define TIME(FUNC) \
  { \
    std::cout << "Timing " << #FUNC << " ...  " << std::flush; \
    std::cout << std::setprecision(5) << timeit(FUNC) << " msec" << std::endl; \
  }

#define milliseconds(x) \
  (std::chrono::duration_cast<std::chrono::nanoseconds>(x).count() / 1e6)
#define timeNow() std::chrono::high_resolution_clock::now()

double timeit(std::function<void()> fn) {
  // warmup
  for (int i = 0; i < 5; ++i) {
    fn();
  }

  int numIters = 100;
  auto start = timeNow();
  for (int i = 0; i < numIters; i++) {
    fn();
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
  std::vector<float> costs(B);
  std::vector<float> alphas(B * T * (U + 1));
  std::vector<float> logNorms(B * T * (U + 1));

  auto transducerForward = [&]() {
      forward(
          emissions.data(),
          predictions.data(),
          costs.data(),
          alphas.data(),
          logNorms.data(),
          labels.data(),
          inputLengths.data(),
          labelLengths.data(),
          B, T, U + 1, V, 0, false);
  };
  TIME(transducerForward);

  std::vector<float> egrads(B * T * V);
  std::vector<float> pgrads(B * (U + 1) * V);
  auto transducerBackward = [&]() {
      backward(
          emissions.data(),
          predictions.data(),
          egrads.data(),
          pgrads.data(),
          alphas.data(),
          logNorms.data(),
          labels.data(),
          inputLengths.data(),
          labelLengths.data(),
          B, T, U + 1, V, 0, false);
  };
  TIME(transducerBackward);
}

int main() {
  int B = 2;
  int T = 100;
  int U = 100;
  int V = 32;
  timeTransducer(B, T, U, V);
}
