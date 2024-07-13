/***********************************************************************************************************************
 * @file    OBPOTF.h
 * @author  Imanol Etxezarreta (ietxezarretam@gmail.com)
 * 
 * @brief   Interface object of a BPOTF decoder, with associated methods. BPOTF decoder uses Kruskal's algorithm and 
 *          Disjoint-Set Advanced Data Structure to offer a quick Low-Density Parity Check syndrome decodification. 
 *          (Revisar)
 * 
 * @version 0.1
 * @date    17/05/2024
 * 
 * @copyright Copyright (c) 2024
 * 
 **********************************************************************************************************************/
#ifndef OBPOTF_H_
#define OBPOTF_H_

// std includes
#include <vector>
#include <inttypes.h>

// Pybind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// Custom includes
#include "CSC/OCSC.h"

#include "../ldpc_v2_src/bp.hpp"

namespace py = pybind11;

class OBPOTF
{
   private:
   float const m_p;  // Error probabilities.
   uint64_t m_u64_pcm_rows;   // Parity check matrix number of rows.
   uint64_t m_u64_pcm_cols;   // Parity check matrix number of columns.
   std::vector<int64_t> m_ai64_idx_matrix; // Row indexes per column where a non-trivial element is located in the pcm.
   uint16_t m_u16_idx_matrix_rows;  // Row number of m_ai64_idx_matrix. Columns will be m_u64_pcm_cols.

   OCSC * m_po_csc_mat = nullptr; // Matrix in Compressed-Sparse-Column format.

   ldpc::bp::BpSparse * m_po_bpsparse = nullptr; // Pointer to the pcm in format for BpDecoder object
   ldpc::bp::BpDecoder * m_po_primary_bp = nullptr; // Pointer to primary BP decoder object.
   ldpc::bp::BpDecoder * m_po_secondary_bp = nullptr; // Pointer to secondary BP decoder in case the first one fails and Kruskal is needed.

   std::vector<uint64_t> m_au64_index_array; // Array that holds indexes from 0 to m_u64_pcm_cols-1 to be sorted

   std::vector<uint64_t> koh_v2_classical_uf(py::array_t<double> const & llrs);

   std::vector<uint64_t> koh_v2_uf(py::array_t<float> const & llrs);

   std::vector<uint64_t> koh_v2_uf_csc(py::array_t<float> const & llrs);
   std::vector<uint64_t> koh_v2_uf_csc(std::vector<double> const & llrs);

   std::vector<uint64_t> sort_indexes(py::array_t<double> const & llrs);

   std::vector<uint64_t *> sort_indexes_nc(py::array_t<double> const & llrs);
   std::vector<uint64_t *> sort_indexes_nc(std::vector<double> const & llrs);

   public:
   /********************************************************************************************************************
    * @brief Construct a new OBPOTF object from the input values.
    * 
    * @param pcm[in] Parity check matrix. It is passed as a py::array_t in Fortran style (Column Major) for speed and
    *                avoid copying the matrix.
    * @param p[in]   Phisical error to initialize the bp_decoder.
    *******************************************************************************************************************/
   OBPOTF(py::array_t<uint8_t, py::array::f_style> const & pcm, float const & p);

   /********************************************************************************************************************
    * @brief Delete default constructor, to avoid empty objects.
    *******************************************************************************************************************/
   OBPOTF() = delete;

   /********************************************************************************************************************
    * @brief Destroy the OBPOTF object.
    *******************************************************************************************************************/
   ~OBPOTF();

   py::array_t<uint8_t> decode(py::array_t<uint8_t, py::array::c_style> syndrome);

   /********************************************************************************************************************
    * @brief Prints the object's member. Developing purposes and testing.
    * 
    *******************************************************************************************************************/
   void print_object(void);
};

#endif // OBPOTF_H_