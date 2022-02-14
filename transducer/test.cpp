#include <algorithm>
#include <cmath>

#include "test.h"
#include "transducer.h"

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

  std::vector<float> costs(batchSize);
  std::vector<float> alphas(batchSize * maxLabelLength * maxInputLength);
  std::vector<float> logNorms(batchSize * maxLabelLength * maxInputLength);

  forward(
      emissions.data(),
      predictions.data(),
      costs.data(),
      alphas.data(),
      logNorms.data(),
      labels.data(),
      inputLengths.data(),
      labelLengths.data(),
      batchSize,
      maxInputLength,
      maxLabelLength,
      alphabetSize,
      blank,
      false);

  checkClose(costs, expectedCosts);

  std::vector<float> egrads(emissions.size());
  std::vector<float> pgrads(predictions.size());
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
      batchSize,
      maxInputLength,
      maxLabelLength,
      alphabetSize,
      blank,
      false);

  checkClose(egrads, expectedEgrads);
  checkClose(pgrads, expectedPgrads);
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
    TEST(tinyTest);
    TEST(smallTest);
    TEST(bigTest);
}
