#include <algorithm>
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
  auto thresh = std::max<float>(rtol * std::max<float>(a, b), atol);
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
