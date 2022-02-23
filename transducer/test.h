#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <iostream>

#define TEST(func) \
  std::cout << "Testing " << #func << "..."; \
  try { \
    func(); \
    std::cout << "passed!" << std::endl; \
  } catch (const std::exception &ex) { \
    std::cout << "FAILED!" << std::endl; \
    if (ex.what() != std::string("")) { \
      std::cout << ex.what() << std::endl; \
    } \
  }

void checkClose(float a, float b, float rtol=1e-6, float atol=1e-5) {
  float inf = std::numeric_limits<float>::infinity();
  if ((a == inf || a == -inf || b == inf || b == -inf) && (a != b)) {
    throw std::runtime_error("");
  }
  auto thresh = std::max<float>(
      rtol * std::max<float>(std::abs(a), std::abs(b)), atol);
  if (std::abs(a - b) > thresh) {
    throw std::runtime_error("");
  }
}

void checkClose(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float rtol=1e-6, float atol=1e-5) {
  if (a.size() != b.size()) {
    throw std::runtime_error("");
  }
  for (int i = 0; i < a.size(); ++i) {
    checkClose(a[i], b[i], rtol, atol);
  }
}

void checkSame(
    const std::vector<int>& a,
    const std::vector<int>& b) {
  if (a != b) {
    throw std::runtime_error("");
  }
}

std::vector<float> computeLogNorms(
    const std::vector<float>& emissions,
    const std::vector<float>& predictions,
    const std::vector<int>& inputLengths,
    const std::vector<int>& labelLengths,
    int maxT, int maxU, int V) {
  int B = inputLengths.size();
  std::vector<float> logNorms(B * maxT * maxU);
  for (int b = 0; b < B; ++b) {
    int T = inputLengths[b];
    int U = labelLengths[b] + 1;
    int eo = b * maxT * V;
    int po = b * maxU * V;
    int lo = b * maxT * maxU;
    for (int t = 0; t < T; ++t) {
      for (int u = 0; u < U; ++u) {
        float maxScore = -std::numeric_limits<float>::infinity();
        for (int v = 0; v < V; ++v) {
          maxScore = std::max(maxScore, emissions[eo + t * V + v] + predictions[po + u * V + v]);
        }
        float expSum = 0.0;
        for (int v = 0; v < V; ++v) {
          expSum += std::exp(emissions[eo + t * V + v] + predictions[po + u * V + v] - maxScore);
        }
        logNorms[lo + t * maxU + u] = std::log(expSum) + maxScore;
      }
    }
  }
  return logNorms;
}

void accumulateGrads(
    const std::vector<float>& emissions,
    const std::vector<float>& predictions,
    std::vector<float>& egrads,
    std::vector<float>& pgrads,
    std::vector<float>& dalphas,
    const std::vector<float>& logNorms,
    const std::vector<int>& inputLengths,
    const std::vector<int>& labelLengths,
    int maxT, int maxU, int V) {
  int B = inputLengths.size();
  for (int b = 0; b < B; ++b) {
    int T = inputLengths[b];
    int U = labelLengths[b] + 1;
    int eo = b * maxT * V;
    int po = b * maxU * V;
    int lo = b * maxT * maxU;

    for (int t = 0; t < T; ++t) {
      for (int u = 0; u < U; ++u) {
        float dalpha = dalphas[lo + t * maxU + u];
        float logNorm = logNorms[lo + t * maxU + u];
        for (int v = 0; v < V; ++v) {
          float score = std::exp(
            emissions[eo + t * V + v] + predictions[po + u * V + v] - logNorm);
          egrads[eo + t * V + v] += dalpha * score;
          pgrads[po + u * V + v] += dalpha * score;
        }
      }
    }
  }
}
