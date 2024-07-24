// Pybind11 header libraries
#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "BPOTF/OBPOTF.h"

namespace py = pybind11;

// Bindings for the bpbp module
PYBIND11_MODULE(BPOTF, BPOTF) {
   py::enum_<OBPOTF::ECodeType_t>(BPOTF, "ECodeType")
      .value("E_GENERIC", OBPOTF::ECodeType_t::E_GENERIC)
      .value("E_CLN", OBPOTF::ECodeType_t::E_CLN)
      .export_values();

   py::class_<OBPOTF>(BPOTF,"OBPOTF")
      .def(py::init<py::object const &, float const &, OBPOTF::ECodeType_t const>(),
            py::arg("pcm"),
            py::arg("p"),
            py::arg("code_type") = OBPOTF::ECodeType_t::E_GENERIC)
      .def("print_object", &OBPOTF::print_object)
      .def("otf_uf", py::overload_cast<py::array_t<double, C_FMT> const &>(&OBPOTF::otf_uf))
      .def("decode", &OBPOTF::decode);

}