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
#include <chrono>

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include <pybind11/embed.h>

#include "OBPOTF.h"
#include "DisjointSet/DisjointSet.h"
#include "CSC/OCSC.h"

using namespace py::literals;

/***********************************************************************************************************************
 * Helper functions
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
:m_p(p)
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
      uint16_t u16_count = 0U;
      std::vector<int64_t> ai64_curr_col;
      for (uint64_t u64_r_idx = 0UL; u64_r_idx < m_u64_pcm_rows; u64_r_idx++)
      {
         uint64_t u64_eff_idx = (u64_c_idx * m_u64_pcm_rows) + u64_r_idx;
         if (au8_pcm_sp[u64_eff_idx] == 1U)
         {
            u16_count++;
            ai64_curr_col.push_back(u64_r_idx);
         }
      }
      
      if (u16_count == 1U)
      {
         if (ai64_curr_col[0] < m_u64_pcm_rows / 2) // Take advantage of the integer division truncation.
         {
            ai64_curr_col.push_back(m_u64_pcm_rows+1);
         }
         else
         {
            ai64_curr_col.push_back(m_u64_pcm_rows+2);
         }
      }
      aai64_temp_vec.push_back(ai64_curr_col);

      if (u16_count > m_u16_idx_matrix_rows)
      {
         m_u16_idx_matrix_rows = u16_count;
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

   // Create CSC type object.
   m_po_csc_mat = new OCSC(au8_pcm_sp, m_u64_pcm_rows);

   for (uint64_t u64_idx = 0; u64_idx < m_u64_pcm_cols; ++u64_idx)
   {
      std::vector<uint64_t> au64_col_checks = m_po_csc_mat->get_col_row_idxs(u64_idx);

      if (au64_col_checks.size() == 1)
      {
         uint64_t u64_check_row = au64_col_checks[0];
         if (u64_check_row > (m_u64_pcm_rows / 2))
         {
            m_po_csc_mat->add_row_idx(u64_idx, m_u64_pcm_rows+1);
         }
         else
         {
            m_po_csc_mat->add_row_idx(u64_idx, m_u64_pcm_rows+2);
         }
      }
   }

   // Initialize the ldpc bp python object
   py::object bp_decoder = py::module_::import("ldpc").attr("bp_decoder");
   m_bpd = bp_decoder(au8_pcm, "error_rate"_a=p);
   m_bpd_secondary = bp_decoder(au8_pcm, "error_rate"_a=p);
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

py::array_t<uint8_t> OBPOTF::decode(py::array_t<int, py::array::c_style> syndrome)
{
   py::array_t<uint8_t> py_recovered_err = m_bpd.attr("decode")(syndrome);
   
   int py_converge = m_bpd.attr("converge").cast<int>();

   if (py_converge != 1)
   {
      py::array_t<double> llrs = m_bpd.attr("log_prob_ratios").cast<py::array_t<double>>();

      //std::vector<uint64_t> columns_chosen = koh_v2_classical_uf(llrs);
      std::vector<uint64_t> columns_chosen = koh_v2_uf(llrs);

      std::vector<float> updated_probs(m_u64_pcm_cols, 0);
      for (uint64_t u64_idx = 0U; u64_idx < columns_chosen.size(); u64_idx++)
         updated_probs[columns_chosen[u64_idx]] = m_p;
      py::array_t<float> py_updated_probs = as_pyarray(std::move(updated_probs));

      //py::array_t<double> & pyref_channel_probs = m_bpd_secondary.attr("channel_probs")().cast<py::array_t<double>>;
      m_bpd_secondary.attr("update_channel_probs")(py_updated_probs);
      py_recovered_err = m_bpd_secondary.attr("decode")(syndrome);
   }

   return py_recovered_err;
}

std::vector<uint64_t> OBPOTF::sort_indexes(py::array_t<double> const & llrs) 
{
   // COPY! Not happy...
   std::vector<uint64_t> idx = m_au64_index_array;

   std::sort(idx.begin(), idx.end(), 
               [&llrs](size_t i1, size_t i2) 
               {
                  return std::less<double>{}(llrs.data()[i1], llrs.data()[i2]);
               }
            );

   return idx;
}

std::vector<uint64_t *> OBPOTF::sort_indexes_nc(py::array_t<double> const & llrs) 
{
   std::vector<uint64_t *> idx(m_au64_index_array.size());
   //std::iota(idx.begin(), idx.end(), m_au64_index_array.begin());
   for (uint64_t u64_idx = 0U; u64_idx < idx.size(); u64_idx++)
      idx[u64_idx] = m_au64_index_array.data() + u64_idx;

   std::sort(idx.begin(), idx.end(), 
               [&llrs](uint64_t * i1, uint64_t * i2) 
               {
                  return std::less<double>{}(llrs.data()[*i1], llrs.data()[*i2]);
               }
            );

   return idx;
}

std::vector<uint64_t> OBPOTF::koh_v2_classical_uf(py::array_t<double> const & llrs)
{
   uint16_t const & hog_rows = m_u16_idx_matrix_rows;
   uint64_t const & hog_cols = m_u64_pcm_cols;
   
   std::vector<size_t> columns_chosen;
   columns_chosen.reserve(hog_cols);

   std::vector<uint64_t> sorted_idxs = sort_indexes(llrs);

   DisjSet clstr_set = DisjSet(m_u64_pcm_rows+2);

   for (size_t col_idx = 0UL; col_idx < sorted_idxs.size(); col_idx++)
   {
      size_t effective_col_idx = sorted_idxs[col_idx];
      size_t col_offset = effective_col_idx * hog_rows;
      std::span<ssize_t> column_sp(m_ai64_idx_matrix.data() + col_offset, hog_rows);

      long int retcode = -1;
      long int root_set = clstr_set.find(column_sp[0]);
      for (size_t nt_elem_idx = 1UL; nt_elem_idx < column_sp.size(); nt_elem_idx++)
      {
         if (column_sp[nt_elem_idx] == -1L)
            break;

         retcode = clstr_set.set_union_opt(root_set, column_sp[nt_elem_idx]);
         if (retcode == -1L)
            break;
         root_set = retcode;
      }
      if (retcode != -1L)
      {
         columns_chosen.push_back(effective_col_idx);
         // if (columns_chosen.size() == rank)
         //    break;
      }
   }

   return columns_chosen;
}

std::vector<uint64_t> OBPOTF::koh_v2_uf(py::array_t<float> const & llrs)
{
   uint16_t const & hog_rows = m_u16_idx_matrix_rows;
   uint64_t const & hog_cols = m_u64_pcm_cols;
   
   std::vector<uint64_t> columns_chosen;
   columns_chosen.reserve(hog_cols);

   //std::vector<uint64_t> sorted_idxs = sort_indexes(llrs);
   std::vector<uint64_t *> sorted_idxs = sort_indexes_nc(llrs);

   DisjSet clstr_set = DisjSet(m_u64_pcm_rows+2); //+2 due to virtual checks

   for (size_t col_idx = 0UL; col_idx < sorted_idxs.size(); col_idx++)
   {
      //size_t effective_col_idx = sorted_idxs[col_idx];
      size_t effective_col_idx = *(sorted_idxs[col_idx]);
      size_t col_offset = effective_col_idx * hog_rows;
      std::span<ssize_t> column_sp(m_ai64_idx_matrix.data() + col_offset, hog_rows);
      
      std::vector<uint8_t> checker(m_u64_pcm_rows+2, 0);
      std::vector<int> depths = {0, -1, 0};
      bool boolean_condition = true;

      for (size_t nt_elem_idx = 0UL; nt_elem_idx < column_sp.size(); nt_elem_idx++)
      {
         if (column_sp[nt_elem_idx] == -1L)
            break;
         int elem_root = clstr_set.find(column_sp[nt_elem_idx]);
         int elem_depth = clstr_set.get_rank(elem_root);

         if (checker[elem_root] == 1U)
         {
            boolean_condition = false;
            break;
         }
         checker[elem_root] = 1U;

         if (elem_depth > depths[1])
         {
            depths = {elem_root, elem_depth, 0};
         }
         if (elem_depth == depths[1])
         {
            depths[2] = 1;
         }
      }

      if (boolean_condition == true)
      {
         for(size_t elem = 0UL; elem < checker.size(); elem++)
         {
            if (checker[elem] == 1U)
               clstr_set.set_parent(elem, depths[0]);
         }
         columns_chosen.push_back(effective_col_idx);
         if (depths[2] == 1)
         {
            clstr_set.increase_rank(depths[0]);
         }
         // if (columns_chosen.size() == rank)
         //    break;
      }
   }

   return columns_chosen;
}

OBPOTF::~OBPOTF(void)
{
   delete m_po_csc_mat;
}