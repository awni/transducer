#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

#include "test.h"
#include "transducer.h"
#include "transducer_cuda.h"

const float kInf = std::numeric_limits<float>::infinity();

float* deviceAlloc(size_t size) {
  float* ptr;
  CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(float) * size));
  return ptr;
}

std::vector<float> hostCopy(const float* dptr, size_t size) {
  std::vector<float> hostVec(size);
  CUDA_CHECK(cudaMemcpy(
        (void*) hostVec.data(),
        (void*) dptr,
        sizeof(float) * size,
        cudaMemcpyDeviceToHost));
  return hostVec;
}

int* deviceCopy(const std::vector<int>& hostVec) {
  int* dptr;
  CUDA_CHECK(cudaMalloc((void**)&dptr, sizeof(float) * hostVec.size()));
  CUDA_CHECK(cudaMemcpy(
      (void*) dptr,
      (void*) hostVec.data(),
      sizeof(int) * hostVec.size(),
      cudaMemcpyHostToDevice));
  return dptr;
}

float* deviceCopy(const std::vector<float>& hostVec) {
  float* dptr = deviceAlloc(hostVec.size());
  CUDA_CHECK(cudaMemcpy(
      (void*) dptr,
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

void testForwardBackward(
    const std::vector<float>& emissions,
    const std::vector<float>& predictions,
    const std::vector<int>& labels,
    const std::vector<int>& inputLengths,
    const std::vector<int>& labelLengths,
    int alphabetSize,
    const std::vector<float>& expectedCosts,
    const std::vector<float>& expectedEgrads,
    const std::vector<float>& expectedPgrads) {

  int maxInputLength = *std::max_element(
      inputLengths.begin(), inputLengths.end());
  int maxLabelLength = *std::max_element(
      labelLengths.begin(), labelLengths.end()) + 1;
  int batchSize = emissions.size() / (alphabetSize * maxInputLength);
  int blank = 0;

  auto emissionsD = deviceCopy(emissions);
  auto predictionsD = deviceCopy(predictions);
  auto labelsD = deviceCopy(labels);
  auto inputLengthsD = deviceCopy(inputLengths);
  auto labelLengthsD = deviceCopy(labelLengths);
  float* costsD = deviceAlloc(batchSize);
  float* alphas = deviceAlloc(batchSize * maxInputLength * maxLabelLength);
  float* logNorms = deviceAlloc(batchSize * maxInputLength * maxLabelLength);

  forward(
      emissionsD,
      predictionsD,
      costsD,
      alphas,
      logNorms,
      labelsD,
      inputLengthsD,
      labelLengthsD,
      batchSize,
      maxInputLength,
      maxLabelLength,
      alphabetSize,
      blank,
      true);

  auto costs = hostCopy(costsD, batchSize);
  checkClose(costs, expectedCosts);

  float* egradsD = deviceAlloc(emissions.size());
  float* pgradsD = deviceAlloc(predictions.size());
  backward(
      emissionsD,
      predictionsD,
      egradsD,
      pgradsD,
      alphas,
      logNorms,
      labelsD,
      inputLengthsD,
      labelLengthsD,
      batchSize,
      maxInputLength,
      maxLabelLength,
      alphabetSize,
      blank,
      true);

  auto egrads = hostCopy(egradsD, emissions.size());
  auto pgrads = hostCopy(pgradsD, predictions.size());
//  checkClose(egrads, expectedEgrads);
//  checkClose(pgrads, expectedPgrads);

  deviceFree(emissionsD);
  deviceFree(predictionsD);
  deviceFree(labelsD);
  deviceFree(inputLengthsD);
  deviceFree(labelLengthsD);
  deviceFree(costsD);
  deviceFree(alphas);
  deviceFree(logNorms);
  deviceFree(egradsD);
  deviceFree(pgradsD);
}

void logNormsTest() {
  auto checkLogNorms = [](
      const std::vector<float>& emissions,
      const std::vector<float>& predictions,
      const std::vector<int>& inputLengths,
      const std::vector<int>& labelLengths,
      int batchSize,
      int alphabetSize,
      int maxT,
      int maxU) {
    auto emissionsD = deviceCopy(emissions);
    auto predictionsD = deviceCopy(predictions);
    auto inputLengthsD = deviceCopy(inputLengths);
    auto labelLengthsD = deviceCopy(labelLengths);
    size_t lNormSize = maxT * maxU * batchSize;
    auto logNormsD = deviceAlloc(lNormSize);
    cuda::computeLogNorms(
        emissionsD,
        predictionsD,
        logNormsD,
        inputLengthsD,
        labelLengthsD,
        batchSize,
        maxT,
        maxU,
        alphabetSize);
    auto lNorms = hostCopy(logNormsD, lNormSize);
    deviceFree(emissionsD);
    deviceFree(predictionsD);
    deviceFree(inputLengthsD);
    deviceFree(labelLengthsD);
    deviceFree(logNormsD);

    for (int b = 0; b < batchSize; ++b) {
      int lIdx = b * maxT * maxU;
      int T = inputLengths[b];
      int U = labelLengths[b] + 1;
      for (int t = 0; t < T; ++t) {
        for (int u = 0; u < U; ++u) {
          float maxScore = -kInf;
          int eIdx = b * maxT * alphabetSize + t * alphabetSize;
          int pIdx = b * maxU * alphabetSize + u * alphabetSize;
          for (int i = 0; i < alphabetSize; ++i) {
            maxScore = std::max(
                maxScore,
                emissions[eIdx + i] + predictions[pIdx + i]);
          }
          float logNorm = 0.0;
          if (maxScore == kInf || maxScore == -kInf) {
            logNorm = maxScore;
          } else {
            for (int i = 0; i < alphabetSize; ++i) {
              logNorm += std::exp(
                  emissions[eIdx + i] + predictions[pIdx + i] - maxScore);
            }
            logNorm = std::log(logNorm) + maxScore;
          }
          auto ln = lNorms[lIdx + t * U + u];
          checkClose(logNorm, ln);
        }
      }
    } 
  };

  {
    // Small test
    std::vector<float> emissions = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> predictions = {1.0, 2.0};
    checkLogNorms(emissions, predictions, {2}, {0}, 1, 2, 2, 1);
  }

  {
    std::vector<float> emissions(4, -kInf);
    std::vector<float> predictions(2, -kInf);
    checkLogNorms(emissions, predictions, {2}, {0}, 1, 2, 2, 1);
  }

  {
    std::vector<float> emissions(4, kInf);
    std::vector<float> predictions(2, kInf);
    checkLogNorms(emissions, predictions, {2}, {0}, 1, 2, 2, 1);
  }
 
  // A few larger tests with variable sizes
  std::vector<int> Bs = {0, 1, 10};
  std::vector<int> Ts = {0, 1, 10, 100};
  std::vector<int> Us = {1, 10, 100};
  std::vector<int> Vs = {0, 8, 20, 32, 101, 128};
  auto randu = []() {
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  };
  for (auto B : Bs) {
  for (auto T : Ts) {
  for (auto U : Us) {
  for (auto V : Vs) {
    std::vector<float> emissions(B * T * V);
    std::generate(emissions.begin(), emissions.end(), randu);
    std::vector<float> predictions(B * U * V);
    std::generate(predictions.begin(), predictions.end(), randu);
    std::vector<int> inputLengths(B);
    std::generate(
        inputLengths.begin(),
        inputLengths.end(),
        [T](){ return std::rand() % (T + 1); });
    std::vector<int> labelLengths(B);
    std::generate(
        labelLengths.begin(),
        labelLengths.end(),
        [U](){ return std::rand() % U; });
    checkLogNorms(
        emissions, predictions, inputLengths, labelLengths, B, V, T, U);
  }
  }
  }
  }
}

void tinyTest() {
  std::vector<float> emissions = {1.0, 2.0, 3.0, 4.0};
  std::vector<float> predictions = {1.0, 2.0};
  int inputLength = 2;
  int labelLength = 0;
  int alphabetSize = 2;

  auto logNorm1 = std::log(
      std::exp(emissions[0] + predictions[0]) +
      std::exp(emissions[1] + predictions[1]));
  auto logNorm2 = std::log(
      std::exp(emissions[2] + predictions[0]) +
      std::exp(emissions[3] + predictions[1]));
  float expectedCost = -(
      emissions[0] + predictions[0] - logNorm1 +
      emissions[2] + predictions[0] - logNorm2);

  auto norm1 = std::exp(logNorm1);
  auto norm2 = std::exp(logNorm2);
  std::vector<float> expectedEgrads =
    {std::exp(emissions[0] + predictions[0]) / norm1 - 1,
     std::exp(emissions[1] + predictions[1]) / norm1,
     std::exp(emissions[2] + predictions[0]) / norm2 - 1,
     std::exp(emissions[3] + predictions[1]) / norm2};
  std::vector<float> expectedPgrads =
    {expectedEgrads[0] + expectedEgrads[2],
     expectedEgrads[1] + expectedEgrads[3]};

  testForwardBackward(
      emissions,
      predictions,
      {},
      {inputLength},
      {labelLength},
      alphabetSize,
      {expectedCost},
      expectedEgrads,
      expectedPgrads);
}

void smallTest() {
  std::vector<float> emissions =
    {0.1, 0.6, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.2, 0.1, 0.1};

  std::vector<float> predictions =
    {0.1, 0.6, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.6, 0.1, 0.1,
     0.1, 0.1, 0.2, 0.8, 0.1};
  std::vector<int> labels = {1, 2};
  int inputLength = 2;
  int labelLength = 2;
  int alphabetSize = 5;

  float expectedCost = 4.843925;
  std::vector<float> expectedEgrads =
    {-0.6884234547615051, -0.05528555065393448, 0.0728120282292366, 0.3593204915523529, 0.3115764856338501,
     -0.6753658056259155, 0.08346927165985107, -0.21995842456817627, 0.48722079396247864, 0.3246341943740845};
  std::vector<float> expectedPgrads =
    {-0.07572315633296967, -0.5175057649612427, 0.2010551244020462, 0.19608692824840546, 0.19608692824840546,
     -0.1768123358488083, 0.30765989422798157, -0.5961406826972961, 0.23264653980731964, 0.23264653980731964,
     -1.1112537384033203, 0.2380295693874359, 0.24793916940689087, 0.41780781745910645, 0.20747721195220947};

  testForwardBackward(
      emissions,
      predictions,
      labels,
      {inputLength},
      {labelLength},
      alphabetSize,
      {expectedCost},
      expectedEgrads,
      expectedPgrads);
}

void bigTest() {
  // batchSize x maxInputLength x alphabetSize
  std::vector<float> emissions =
    {0.8764081559029704, 0.8114401931890338, 0.6508828493896047,
     0.6831969720272136, 0.794939425350507, 0.4771495462110181,
     0.07800002444603382, 0.007794919225017516, 0.9478301043860103,
     0.49619506263326396, 0.7345710606552497, 0.7741700701082916,

     0.7084607475161292, 0.9860726712179101, 0.7902338818255793,
     0.7691063457590045, 0.5448267745331934, 0.22524027048482376,
     0.2291088288701465, 0.7524300104847589, 0.7273355024795244,
     0.33155408518920104, 0.8068789770558062, 0.6188633401048291};

  // batchSize x maxLabelLength x alphabetSize
  std::vector<float> predictions =
    {0.6223532638505989, 0.3002940148933876, 0.7404674033386307,
     0.01823584315362603, 0.034963374948701054, 0.34892745941957193,
     0.5718051448658747, 0.28205981250440926, 0.7283146324887043,

     0.7755842032974967, 0.5521231124815825, 0.8577769985498179,
     0.42450076602299125, 0.9417870425381804, 0.0072059916072961805,
     0.37187505831579304, 0.960974111779922, 0.04504344671276461};

  std::vector<int> labels = {1, 2, 1, 1};
  std::vector<int> inputLengths = {4, 4};
  std::vector<int> labelLengths = {2, 2};
  int alphabetSize = 3;

  std::vector<float> expectedCosts = {4.718404769897461, 4.803375244140625};
  std::vector<float> expectedEgrads =
    {-0.4596531093120575, 0.041041433811187744, 0.4186115860939026,
     -0.4770655333995819, 0.13196370005607605, 0.34510183334350586,
     -0.6760067939758301, 0.09430177509784698, 0.5817050337791443,
     -0.5915795564651489, 0.29016029834747314, 0.3014192581176758,

     -0.5917761325836182, 0.15546470880508423, 0.4363115429878235,
     -0.4406549036502838, 0.14964917302131653, 0.2910056710243225,
     -0.6741735935211182, 0.23876483738422394, 0.4354088008403778,
     -0.6422789096832275, 0.2854732275009155, 0.356805682182312};

  std::vector<float> expectedPgrads =
    {-0.3262518346309662, -0.46784698963165283, 0.7940987944602966,
     -0.429027259349823, 0.5465580821037292, -0.11753084510564804,
     -1.4490258693695068, 0.4787561297416687, 0.9702697992324829,

     -0.5165280699729919, -0.28539586067199707, 0.8019239902496338,
     -0.4294244050979614, 0.07082393765449524, 0.3586004972457886,
     -1.4029310941696167, 1.0439238548278809, 0.35900723934173584};

  testForwardBackward(
      emissions,
      predictions,
      labels,
      inputLengths,
      labelLengths,
      alphabetSize,
      expectedCosts,
      expectedEgrads,
      expectedPgrads);
}

int main() {
  TEST(logNormsTest);
  TEST(tinyTest);
  TEST(smallTest);
  TEST(bigTest);
}
