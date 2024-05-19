/***********************************************************************************************************************
 * @file    OBPOTF.cpp
 * @author  Imanol Etxezarreta (ietxezarretam@gmail.com)
 * 
 * @brief   Implementation of the object and its methods for OBPOTF.
 * 
 * @version 0.1
 * @date    17/05/2024
 * 
 * @copyright Copyright (c) 2024
 * 
 **********************************************************************************************************************/

#include <numeric>
#include <span>
#include <iostream>

#include <pybind11/embed.h>

#include "OBPOTF.h"

using namespace py::literals;

/***********************************************************************************************************************
 * Private functions
 **********************************************************************************************************************/

// helper function to avoid making a copy when returning a py::array_t
// author: https://github.com/YannickJadoul
// source: https://github.com/pybind/pybind11/issues/1042#issuecomment-642215028
template <typename Sequence>
inline static py::array_t<typename Sequence::value_type> as_pyarray(Sequence &&seq) {
   auto size = seq.size();
   auto data = seq.data();
   std::unique_ptr<Sequence> seq_ptr = std::make_unique<Sequence>(std::move(seq));
   auto capsule = py::capsule(seq_ptr.get(), 
                              [](void *p) 
                              {
                                 std::unique_ptr<Sequence>(reinterpret_cast<Sequence *>(p));
                              }
                              );
   seq_ptr.release();

   return py::array(size, data, capsule);
}

/**
 * \brief Returns span<T> from py:array_T<T>. Efficient as zero-copy.
 * \tparam T Type.
 * \param passthrough Numpy array.
 * \return Span<T> that with a clean and safe reference to contents of Numpy array.
 */
template<typename T>
inline static std::span<T> toSpan2D(py::array_t<T, py::array::f_style> const & passthrough)
{
	py::buffer_info passthroughBuf = passthrough.request();
	if (passthroughBuf.ndim != 2) {
		throw std::runtime_error("Error. Number of dimensions must be two");
	}
	size_t length = passthroughBuf.shape[0] * passthroughBuf.shape[1];
	T* passthroughPtr = static_cast<T*>(passthroughBuf.ptr);
	std::span<T> passthroughSpan(passthroughPtr, length);
	return passthroughSpan;
}

OBPOTF::OBPOTF(py::array_t<uint8_t, py::array::f_style> const & au8_pcm, 
               float const & p)
{
   py::buffer_info py_pcm_bufinfo = au8_pcm.request();

   if (py_pcm_bufinfo.ndim != 2)
   {
      throw std::runtime_error("[ERROR] The pcm must be of ndim = 2!");
   }

   m_u64_pcm_rows = au8_pcm.shape(0L);
   m_u64_pcm_cols = au8_pcm.shape(1L);

   // Form the vector of indexes to be sorted
   m_au64_index_array = std::vector<uint64_t>(m_u64_pcm_cols);
   std::iota(m_au64_index_array.begin(), m_au64_index_array.end(), 0UL);

   // Span pointer to the pcm python input to be faster
   std::span<uint8_t> const au8_pcm_sp = toSpan2D(au8_pcm);
   // Temporal matrix that holds the row indexes per column
   std::vector<std::vector<int64_t>> aai64_temp_vec;
   // Initialize index matrix rows.
   m_u16_idx_matrix_rows = 0U;

   // Iterate through the pcm to get the non-trivial row indexes and maximum number of nt per column.
   for (uint64_t u64_c_idx = 0UL; u64_c_idx < m_u64_pcm_cols; u64_c_idx++)
   {
      uint8_t u8_count = 0U;
      std::vector<int64_t> ai64_curr_col;
      for (uint64_t u64_r_idx = 0UL; u64_r_idx < m_u64_pcm_rows; u64_r_idx++)
      {
         uint64_t u64_eff_idx = (u64_c_idx * m_u64_pcm_rows) + u64_r_idx;
         if (au8_pcm_sp[u64_eff_idx] == 1U)
         {
            u8_count++;
            ai64_curr_col.push_back(u64_r_idx);
         }
      }
      
      if (u8_count == 1U)
      {
         if (u8_count < m_u64_pcm_rows / 2) // Take advantage of the integer division truncation.
         {
            ai64_curr_col.push_back(m_u64_pcm_rows+1);
         }
         else
         {
            ai64_curr_col.push_back(m_u64_pcm_rows+2);
         }
      }
      aai64_temp_vec.push_back(ai64_curr_col);

      if (u8_count > m_u16_idx_matrix_rows)
      {
         m_u16_idx_matrix_rows = u8_count;
      }
   }

   // Initialize the index_matrix to be idx_matrix_rows x pcm_cols of -1s, and fill it in the loop
   m_ai64_idx_matrix = std::vector<int64_t>(m_u64_pcm_cols*m_u16_idx_matrix_rows, -1L);
   for (uint64_t u64_idx = 0U; u64_idx < aai64_temp_vec.size(); u64_idx++)
   {
      for (uint64_t u64_idx2 = 0U; u64_idx2 < aai64_temp_vec[u64_idx].size(); u64_idx2++)
      {
         m_ai64_idx_matrix[u64_idx * m_u16_idx_matrix_rows + u64_idx2] = aai64_temp_vec[u64_idx][u64_idx2];
      }
   }

   // Initialize the ldpc bp python object
   py::object bp_decoder = py::module_::import("ldpc").attr("bp_decoder");
   m_bpd = bp_decoder(au8_pcm, "error_rate"_a=p);
}

void OBPOTF::print_object(void)
{
   std::cout << "m_u64_pcm_rows: " << m_u64_pcm_rows << std::endl;
   std::cout << "m_u64_pcm_cols: " << m_u64_pcm_cols << std::endl;
   std::cout << "m_ai64_idx_matrix: " << std::endl;
   for (uint64_t u64_idx = 0U; u64_idx < m_u16_idx_matrix_rows; u64_idx++)
   {
      for (uint64_t u64_idx2 = 0U; u64_idx2 < m_u64_pcm_cols; u64_idx2++)
      {
         std::cout << m_ai64_idx_matrix[u64_idx2*m_u16_idx_matrix_rows+u64_idx] << " ";
      }
      std::cout << std::endl;
   }

   std::cout << "m_au64_index_array: " << std::endl;
   for (uint64_t u64_idx = 0U; u64_idx < m_u64_pcm_cols; u64_idx++)
      std::cout << m_au64_index_array[u64_idx] << " ";
   std::cout << std::endl;


}