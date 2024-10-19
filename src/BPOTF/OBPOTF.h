/***********************************************************************************************************************
 * @file    OBPOTF.h
 * @author  Imanol Etxezarreta (ietxezarretam@gmail.com)
 * 
 * @brief   Object of a BPOTF decoder, with associated methods. BPOTF decoder uses Kruskal's algorithm and 
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

#define C_FMT py::array::c_style
#define F_FMT py::array::f_style

class OBPOTF
{
   /******************************************************************************************************************* 
    * PRIVATE MEMBER VARIABLE DECLARATION 
    *******************************************************************************************************************/
   private:

   //! Error probabilities.
   float const m_p;
   //! Parity check matrix number of rows.
   uint64_t m_u64_pcm_rows;
   //! Parity check matrix number of columns.
   uint64_t m_u64_pcm_cols;

   // TODO: Rename to specify explicitly that it is the pcm matrix
   //! Matrix in Compressed-Sparse-Column format.
   OCSC * m_po_csc_mat = nullptr;

   //! Transfer matrix in case a DEM is provided.
   OCSC * m_po_transfer_csc_mat = nullptr;

   //! Pointer to the pcm in format for BpDecoder object
   ldpc::bp::BpSparse * m_po_bpsparse = nullptr;
   //! Pointer to primary BP decoder object.
   ldpc::bp::BpDecoder * m_po_primary_bp = nullptr;
   //! Pointer to secondary BP decoder in case the first one fails and Kruskal is needed. 
   ldpc::bp::BpDecoder * m_po_secondary_bp = nullptr;

   //! Array that holds indexes from 0 to m_u64_pcm_cols-1 to be sorted.
   std::vector<uint64_t> m_au64_index_array;

   //! Callback to the decoding method selected when constructing the object.
   py::array_t<uint8_t> (OBPOTF::*m_pf_decoding_func)(py::array_t<uint8_t, C_FMT> const &);

   /********************************************************************************************************************
    * PUBLIC MEMBER VARIABLE DECLARATION 
    *******************************************************************************************************************/
   public:

   /********************************************************************************************************************
    * @typedef ENoiseType_t
    * @brief   This typedef holds the different noise models that can be passed to the decoder. Depending on the type
    *          passed, the decoder could also use (by input argument or trying to obtaining it) a transference matrix
    *          to simplify the decoding procedure.
    *******************************************************************************************************************/
   typedef enum 
   {
      E_CC     = 0,  //!< Code Capacity (Default mode).
      E_PHEN   = 1,  //!< Phenomenological.
      E_CLN    = 2   //!< Circuit-level noise.
   } ENoiseType_t;

   /********************************************************************************************************************
    * PRIVATE CLASS METHOD DECLARATION
    *******************************************************************************************************************/
   private:

   /********************************************************************************************************************
    * @brief This routine performs the OTF algorithm using the clasical Unified-Find method.
    * 
    * @param llrs[in]   The llrs is a vector containing the probabilities of the PCM columns.
    * @return std::vector<uint64_t> The return value is a vector containing the recovered error.
    *******************************************************************************************************************/
   std::vector<uint64_t> otf_classical_uf(std::vector<double> const & llrs);

   /********************************************************************************************************************
    * @brief This routine performs the OTF algorithm.
    * 
    * @param llrs[in]   The llrs is a vector containing the probabilities of the PCM columns.
    * @return std::vector<uint64_t> The return value is a vector containing the recovered error.
    *******************************************************************************************************************/
   std::vector<uint64_t> otf_uf(std::vector<double> const & llrs);

   /********************************************************************************************************************
    * @brief This routine returns a vector of the sorted indexes based on the probabilities of the llrs. It copies the 
    *        initial vector with the unsorted indexes from the member variable m_au64_index_array.
    * 
    * @param llrs[in]   The llrs is a vector containing the probabilities of the PCM columns.
    * @return std::vector<uint64_t> The return variable is a vector with the sorted indexes according the llrs.
    *******************************************************************************************************************/
   std::vector<uint64_t> sort_indexes(py::array_t<double> const & llrs);

   /********************************************************************************************************************
    * @brief This routine returns a sorted vector of pointers to the sorted indexes stored in the member variable
    *        m_au64_index_array, based on the probabilities in the llrs.
    * 
    * @param llrs[in]   The llrs is a vector containing the probabilities of the PCM columns.
    * @return std::vector<uint64_t *> The return variable is a sorted vector of pointers to the indexes based on the
    *                                 llrs.
    *******************************************************************************************************************/
   std::vector<uint64_t *> sort_indexes_nc(std::vector<double> const & llrs);

   /********************************************************************************************************************
    * @brief This routine returns a sorted vector of pointers to the sorted indexes stored in the member variable
    *        m_au64_index_array, based on the probabilities in the llrs.
    * 
    * @param llrs[in]   The llrs is a span containing the probabilities of the PCM columns.
    * @return std::vector<uint64_t *> The return variable is a sorted vector of pointers to the indexes based on the
    *                                 llrs.
    *******************************************************************************************************************/
   std::vector<uint64_t *> sort_indexes_nc(std::span<double> const & llrs);

   /********************************************************************************************************************
    * @brief This routine executes a generic decode procedure, which is done for surface-codes. It is registered as a 
    *        callback in the member variable m_pf_decoding_func when the object is created if the enumeration type is
    *        set to E_GENERIC.
    * 
    * @param syndrome[in]  A python array in c-style format that indicates the syndrome from which recover the error.
    * @return py::array_t<uint8_t> Output python array with the resulting recovered error.
    *******************************************************************************************************************/
   py::array_t<uint8_t> generic_decode(py::array_t<uint8_t, C_FMT> const & syndrome);

   /********************************************************************************************************************
    * @brief This routine executes the decoding process for circuit-level noise type of errors. It is registered as a 
    *        callback in the member variable m_pf_decoding_func when the object is created if the enumeration type is
    *        set to E_CLN.
    * 
    * @param syndrome[in]  A python array in c-style format that indicates the syndrome from which recover the error.
    * @return py::array_t<uint8_t> Output python array with the resulting recovered error.
    *******************************************************************************************************************/
   py::array_t<uint8_t> cln_decode(py::array_t<uint8_t, C_FMT> const & syndrome);

   /********************************************************************************************************************
    * PUBLIC CLASS METHOD DECLARATION
    *******************************************************************************************************************/
   public:

   /********************************************************************************************************************
    * @brief Construct a new OBPOTF object from the input values. It calls other sub-routines depending on the python 
    *        object that is passed as a parameter.
    * 
    * @param pcm[in]       Parity check matrix. It is passed as a py::object for speed and avoid copying the matrix.
    * @param p[in]         Phisical error to initialize the bp_decoder.
    * @param code_type[in] Type of the error source.
    *******************************************************************************************************************/
   OBPOTF(py::object const & pcm, float const & p, ENoiseType_t const code_type, py::object const * const transfer_mat);

   /********************************************************************************************************************
    * @brief Sub-routine that is called from the object constructor if it is called with a numpy array. It initialized 
    *        the object members from input parameters and executes necessary pre-processings.
    * 
    * @param pcm[in] Parity-check matrix from which to initialize the members.
    *******************************************************************************************************************/
   void OBPOTF_init_from_numpy(py::array_t<uint8_t, F_FMT> const & pcm);
   
   /********************************************************************************************************************
    * @brief Sub-routine that is called from the object constructor if it is called with a scipy_csc object. In this 
    *        case, the object is converted to a pyarray and the the OBPOTF_init_from_numpy is called with it. 
    * 
    * @param pcm[in] Parity-check matrix from which to initialize the members.
    *******************************************************************************************************************/
   void OBPOTF_init_from_scipy_csc(py::object const & pcm);

   /********************************************************************************************************************
    * @brief Delete default constructor, to avoid empty objects.
    *******************************************************************************************************************/
   OBPOTF() = delete;

   /********************************************************************************************************************
    * @brief Destroy the OBPOTF object to free allocated memory.
    *******************************************************************************************************************/
   ~OBPOTF();

   /********************************************************************************************************************
    * @brief This routine performs the OTF algorithm.
    * 
    * (It is public right now because Ton needed only this part of the algorithm for some tests)
    * 
    * @param llrs[in]   The llrs is a vector containing the probabilities of the PCM columns.
    * @return py::array_t<uint64_t> The return value is a py::array_t containing the recovered error.
    *******************************************************************************************************************/
   py::array_t<uint64_t> otf_uf(py::array_t<double, C_FMT> const & llrs);

   /********************************************************************************************************************
    * @brief This is the main decoding routine. This routine calls the registered callback function in 
    *        m_pf_decoding_func with the syndrome to recover the error.
    * 
    * @param syndrome[in]  A python array in c-style format that indicates the syndrome from which to recover the error.
    * @return py::array_t<uint8_t> Returned value is a python array with the recovered error.
    *******************************************************************************************************************/
   py::array_t<uint8_t> decode(py::array_t<uint8_t, C_FMT> const & syndrome);

   /********************************************************************************************************************
    * @brief Prints the object's member. Developing purposes and testing.
    *******************************************************************************************************************/
   void print_object(void);
};

#endif // OBPOTF_H_