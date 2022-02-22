#include <pybind11/pybind11.h>

#include "transducer.h"

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_transducer, m) {
  m.def(
      "forward",
      [](const std::uintptr_t emissions,
         const std::uintptr_t predictions,
         const std::uintptr_t costs,
         const std::uintptr_t alphas,
         const std::uintptr_t logNorms,
         const std::uintptr_t labels,
         const std::uintptr_t input_lengths,
         const std::uintptr_t label_lengths,
         int batch_size,
         int max_input_length,
         int max_label_length,
         int alphabet_size,
         int blank,
         bool use_cuda) {
        forward(
            reinterpret_cast<float*>(emissions),
            reinterpret_cast<float*>(predictions),
            reinterpret_cast<float*>(costs),
            reinterpret_cast<float*>(alphas),
            reinterpret_cast<float*>(logNorms),
            reinterpret_cast<int*>(labels),
            reinterpret_cast<int*>(input_lengths),
            reinterpret_cast<int*>(label_lengths),
            batch_size,
            max_input_length,
            max_label_length,
            alphabet_size,
            blank,
            use_cuda);
      },
      "emissions"_a,
      "predictions"_a,
      "costs"_a,
      "alphas"_a,
      "logNorms"_a,
      "labels"_a,
      "input_lengths"_a,
      "label_lengths"_a,
      "batch_size"_a,
      "max_input_length"_a,
      "max_label_length"_a,
      "alphabet_size"_a,
      "blank"_a,
      "use_cuda"_a);

  m.def(
      "backward",
      [](const std::uintptr_t emissions,
         const std::uintptr_t predictions,
         std::uintptr_t egrads,
         std::uintptr_t pgrads,
         std::uintptr_t lngrads,
         const std::uintptr_t alphas,
         const std::uintptr_t logNorms,
         const std::uintptr_t labels,
         const std::uintptr_t input_lengths,
         const std::uintptr_t label_lengths,
         int batch_size,
         int max_input_length,
         int max_label_length,
         int alphabet_size,
         int blank,
         bool use_cuda) {
        backward(
            reinterpret_cast<float*>(emissions),
            reinterpret_cast<float*>(predictions),
            reinterpret_cast<float*>(egrads),
            reinterpret_cast<float*>(pgrads),
            reinterpret_cast<float*>(lngrads),
            reinterpret_cast<float*>(alphas),
            reinterpret_cast<float*>(logNorms),
            reinterpret_cast<int*>(labels),
            reinterpret_cast<int*>(input_lengths),
            reinterpret_cast<int*>(label_lengths),
            batch_size,
            max_input_length,
            max_label_length,
            alphabet_size,
            blank,
            use_cuda);
      },
      "emissions"_a,
      "predictions"_a,
      "egrads"_a,
      "pgrads"_a,
      "lngrads"_a,
      "alphas"_a,
      "logNorms"_a,
      "labels"_a,
      "input_lengths"_a,
      "label_lengths"_a,
      "batch_size"_a,
      "max_input_length"_a,
      "max_label_length"_a,
      "alphabet_size"_a,
      "blank"_a,
      "use_cuda"_a);

  m.def(
      "viterbi",
      [](const std::uintptr_t emissions,
         const std::uintptr_t predictions,
         std::uintptr_t labels,
         const std::uintptr_t input_lengths,
         const std::uintptr_t label_lengths,
         int batch_size,
         int max_input_length,
         int max_label_length,
         int alphabet_size,
         int blank,
         bool use_cuda) {
        viterbi(
            reinterpret_cast<float*>(emissions),
            reinterpret_cast<float*>(predictions),
            reinterpret_cast<int*>(labels),
            reinterpret_cast<int*>(input_lengths),
            reinterpret_cast<int*>(label_lengths),
            batch_size,
            max_input_length,
            max_label_length,
            alphabet_size,
            blank,
            use_cuda);
      },
      "emissions"_a,
      "predictions"_a,
      "labels"_a,
      "input_lengths"_a,
      "label_lengths"_a,
      "batch_size"_a,
      "max_input_length"_a,
      "max_label_length"_a,
      "alphabet_size"_a,
      "blank"_a,
      "use_cuda"_a);
#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
