// Pybind11 header libraries
#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "BPOTF/OBPOTF.h"

namespace py = pybind11;

// Bindings for the bpbp module
PYBIND11_MODULE(BPOTF, BPOTF) {
   py::class_<OBPOTF>(BPOTF,"OBPOTF")
      .def(py::init<py::object const &, float const &>())
      .def("print_object", &OBPOTF::print_object)
      .def("otf_uf", py::overload_cast<py::array_t<double, C_FMT> const &>(&OBPOTF::otf_uf))
      .def("decode", &OBPOTF::decode);
}