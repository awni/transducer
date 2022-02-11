#include <pybind11/pybind11.h>

#include "transducer.h"

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_transducer, m) {
  m.def(
      "forward",
      forward,
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
      backward,
      "emissions"_a,
      "predictions"_a,
      "egrads"_a,
      "pgrads"_a,
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
#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
