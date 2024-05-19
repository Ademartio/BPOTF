/***********************************************************************************************************************
 * @file    DisjointSet.h
 * @author  Imanol Etxezarreta (ietxezarretam@gmail.com)
 * 
 * @brief   C++ interface of the disjoint-set DSA class. Taken from 
 *          https://www.geeksforgeeks.org/introduction-to-disjoint-set-data-structure-or-union-find-algorithm/ and may
 *          be subject to some changes depending the necessities of bpbp.
 * 
 * @version 0.1
 * @date    08/05/2024
 * 
 * @copyright Copyright (c) 2024
 * 
 **********************************************************************************************************************/
#ifndef DISJOINT_SET_H_
#define DISJOINT_SET_H_

class DisjSet
{
   private:
      int *rank;
      long int *parent;
      long int n; 

   public: 
      // Constructor to create and 
      // initialize sets of n items 
      DisjSet(long int const & n);

      // Creates n single item sets 
      void make_set();

      // Finds set of given item x 
      long int find(long int const & x);

      // Sets the parent of element x to new_parent.
      void set_parent(long int const & x, long int const & new_parent);

      // Returns the rank associated with the x value.
      int get_rank(long int const & x);

      // Increases by 1 the rank of the element x.
      void increase_rank(long int const & x);

      // Do union of two sets by rank represented 
      // by x and y. 
      int set_union(long int const & x, long int const & y);

      // Do the union of two sets by rank in a UF manner
      long int set_union_opt(long int const & parent, 
                              long int const & candidate);
};

#endif // DISJOINT_SET_H_