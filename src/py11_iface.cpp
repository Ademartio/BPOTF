// Pybind11 header libraries
#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "BPOTF/OBPOTF.h"

namespace py = pybind11;

// Bindings for the bpbp module
PYBIND11_MODULE(BPOTF, BPOTF) {
   py::class_<OBPOTF>(BPOTF,"OBPOTF")
      .def(py::init<py::array_t<uint8_t, py::array::f_style> const &, float const &>())
      .def("print_object", &OBPOTF::print_object)
      .def("decode", &OBPOTF::decode);
}