/***********************************************************************************************************************
 * @file    py11_iface.cpp
 * @author  Imanol Etxezarreta (ietxezarretam@gmail.com)
 * 
 * @brief   This file is an interface with Pybind11 to expose the BPOTF class and its methods to a python module.
 * 
 * @version 0.1
 * @date    21/08/2024
 * 
 * @copyright Copyright (c) 2024
 * 
 **********************************************************************************************************************/

// Pybind11 header libraries
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// Custom headers
#include "BPOTF/OBPOTF.h"

namespace py = pybind11;

// Bindings for the BPOTF module
PYBIND11_MODULE(BPOTF, BPOTF) {
   // Export enumeration typedef for different error sources supported
   py::enum_<OBPOTF::ECodeType_t>(BPOTF, "ECodeType")
      .value("E_GENERIC", OBPOTF::ECodeType_t::E_GENERIC)
      .value("E_CLN", OBPOTF::ECodeType_t::E_CLN)
      .export_values();

   // Export class and its public methods.
   py::class_<OBPOTF>(BPOTF,"OBPOTF")
      .def(py::init<py::object const &, float const &, OBPOTF::ECodeType_t const>(),
            py::arg("pcm"),   // Parity-check matrix parameter
            py::arg("p"),     // Physical error probability
            py::arg("code_type") = OBPOTF::ECodeType_t::E_GENERIC)   // Error source type. Default: E_GENERIC
      .def("print_object", &OBPOTF::print_object)
      .def("otf_uf", py::overload_cast<py::array_t<double, C_FMT> const &>(&OBPOTF::otf_uf))
      .def("decode", &OBPOTF::decode);

}