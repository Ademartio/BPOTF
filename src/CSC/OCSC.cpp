/***********************************************************************************************************************
 * @file    OCSC.cpp
 * @author  Imanol Etxezarreta (ietxezarretam@gmail.com)
 * 
 * @brief   
 * 
 * @version 0.1
 * @date    28/05/2024
 * 
 * @copyright Copyright (c) 2024
 **********************************************************************************************************************/

#include <iostream>
#include <cstring>
#include <algorithm>

#include "OCSC.h"

OCSC::OCSC(OCSC const & csc_mat):
   m_u64_nnz(csc_mat.m_u64_nnz),
   m_u64_m(csc_mat.m_u64_m),
   m_u64_n(csc_mat.m_u64_n)
{
   m_pu64_r_indices = new uint64_t[m_u64_n+1];
   m_pu64_indptr = new uint64_t[m_u64_nnz];

   std::memcpy(m_pu64_r_indices, csc_mat.m_pu64_r_indices, (m_u64_n+1) * sizeof(uint64_t));
   std::memcpy(m_pu64_indptr, csc_mat.m_pu64_indptr, m_u64_nnz * sizeof(uint64_t));
}

OCSC::OCSC(std::vector<uint64_t> const & u64_nnz_cols_vec,
            std::vector<uint64_t> const & u64_row_idxs_vec,
            uint64_t const & u64_nnz):
            m_u64_nnz(u64_nnz)
{
   m_u64_n = u64_nnz_cols_vec.size() - 1;
   // I think it is not real if the last column is all 0s but al least is something
   m_u64_m = *std::max_element(u64_row_idxs_vec.begin(), u64_row_idxs_vec.end()) + 1;

   m_pu64_r_indices = new uint64_t[m_u64_n+1];
   m_pu64_indptr = new uint64_t[m_u64_nnz];

   std::memcpy(m_pu64_r_indices, u64_nnz_cols_vec.data(), (m_u64_n+1) * sizeof(uint64_t));
   std::memcpy(m_pu64_indptr, u64_row_idxs_vec.data(), m_u64_nnz * sizeof(uint64_t));
}

OCSC::OCSC(std::vector<std::vector<uint8_t>> const & pcm)
{
   uint64_t const u64_m = pcm.size(); // rows
   uint64_t const u64_n = (u64_m > 0) ? pcm[0].size() : 0U; // cols

   m_u64_m = u64_m;
   m_u64_n = u64_n;
   m_u64_nnz = 0U;

   m_pu64_r_indices = new uint64_t[u64_n+1];
   m_pu64_r_indices[0] = 0;

   // count number of non-zero elements
   for (uint64_t j = 0U; j < u64_n; j++)
   {
      for (uint64_t i = 0U; i < u64_m; i++)
      {
         if (pcm[i][j] != 0U)
         {
            m_u64_nnz++;
         }
      }
      m_pu64_r_indices[j+1] = m_u64_nnz;
   }

   // allocate memory
   m_pu64_indptr = new uint64_t[m_u64_nnz];

   // fill array
   uint64_t u64_cont = 0U;
   for (uint64_t j = 0U; j < u64_n; j++)
   {
      for (uint64_t i = 0U; i < u64_m; i++)
      {
         if (pcm[i][j] != 0U)
         {
            m_pu64_indptr[u64_cont] = i;
            u64_cont++;
         }
      }
   }
}

OCSC::OCSC(std::vector<uint8_t> const & pcm, uint64_t const & u64_row_num)
{
   if (pcm.size() % u64_row_num != 0)
   {
      throw std::runtime_error("Error. Number of rows must be equal in all columns.");
   }

   m_u64_m = u64_row_num;
   m_u64_n = pcm.size() / u64_row_num;
   m_u64_nnz = 0U;

   m_pu64_r_indices = new uint64_t[m_u64_n+1];
   m_pu64_r_indices[0] = 0;

   // count number of non-zero elements
   for (uint64_t j = 0U; j < m_u64_n; j++)
   {
      for (uint64_t i = 0U; i < m_u64_m; i++)
      {
         if (pcm[(j*m_u64_m)+i] != 0U)
         {
            m_u64_nnz++;
         }
      }
      m_pu64_r_indices[j+1] = m_u64_nnz;
   }

   // allocate memory
   m_pu64_indptr = new uint64_t[m_u64_nnz];

   // fill array
   uint64_t u64_cont = 0U;
   for (uint64_t j = 0U; j < m_u64_n; j++)
   {
      for (uint64_t i = 0U; i < m_u64_m; i++)
      {
         if (pcm[(j*m_u64_m)+i] != 0U)
         {
            m_pu64_indptr[u64_cont] = i;
            u64_cont++;
         }
      }
   }
}

OCSC::~OCSC()
{
   delete [] m_pu64_r_indices;
   delete [] m_pu64_indptr;
}

void OCSC::print_csc(void)
{
   std::cout << "Shape (MxN): " << m_u64_m << "x" << m_u64_n << std::endl;
   std::cout << "m_u64_nnz: " << m_u64_nnz << std::endl;
   std::cout << "m_pu64_indptr: [";
   for (uint64_t i = 0; i < m_u64_nnz; i++)
   {
      char * str = ", ";
      if (i == m_u64_nnz-1)
         str = "";
      std::cout << m_pu64_indptr[i] << str;
   }
   std::cout << "]\n";

   std::cout << "m_pu64_r_indices: [";
   for (uint64_t i = 0; i < m_u64_n+1; i++)
   {
      char * str = ", ";
      if (i == m_u64_n)
         str = "";
      std::cout << m_pu64_r_indices[i] << str;
   }
   std::cout << "]\n";

}

std::vector<std::vector<uint8_t>> OCSC::expand(void)
{
   std::vector<std::vector<uint8_t>> res_mat(m_u64_m, std::vector<uint8_t>(m_u64_n, 0));

   for (uint64_t i = 1U; i < m_u64_nnz; i++)
   {
      for (uint64_t j = m_pu64_indptr[i-1]; j < m_pu64_indptr[i]; j++)
      {
         res_mat[i][j] = 1U;
      }
   }

   return res_mat;
}