/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing SIMD abstractions for AVX2
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifdef __RVVF__

#ifndef RAJA_policy_tensor_arch_rvv_traits_HPP
#define RAJA_policy_tensor_arch_rvv_traits_HPP

namespace RAJA
{
namespace internal
{
namespace expt
{

#ifndef __LMUL__
#define __LMUL__ 1
#endif
#ifndef __VL__
#define __VL__ 256
#endif
#ifdef __LMUL__
#ifdef __VL__

template<>
struct RegisterTraits<RAJA::expt::rvv_register, int32_t>
{
  using element_type                      = int32_t;
  using register_policy                   = RAJA::expt::rvv_register;
  static constexpr camp::idx_t s_num_bits = __LMUL__ * __VL__  ;
  static constexpr camp::idx_t s_num_elem = __LMUL__ * __VL__ / 32;
  using int_element_type                  = int32_t;
};

template<>
struct RegisterTraits<RAJA::expt::rvv_register, int64_t>
{
  using element_type                      = int64_t;
  using register_policy                   = RAJA::expt::rvv_register;
  static constexpr camp::idx_t s_num_bits = __LMUL__ * __VL__ ;
  static constexpr camp::idx_t s_num_elem =  __LMUL__ * __VL__ / 64;
  using int_element_type                  = int64_t;
};

template<>
struct RegisterTraits<RAJA::expt::rvv_register, float>
{
  using element_type                      = float;
  using register_policy                   = RAJA::expt::rvv_register;
  static constexpr camp::idx_t s_num_bits = __LMUL__ * __VL__ ;
  static constexpr camp::idx_t s_num_elem = __LMUL__ * __VL__ / 32;
  using int_element_type                  = int32_t;
};

template<>
struct RegisterTraits<RAJA::expt::rvv_register, double>
{
  using element_type                      = double;
  using register_policy                   = RAJA::expt::rvv_register;
  static constexpr camp::idx_t s_num_bits = __LMUL__ * __VL__ ;
  static constexpr camp::idx_t s_num_elem = __LMUL__ * __VL__ / 64;
  using int_element_type                  = int64_t;
};





#endif // __VL__
#endif // __LMUL__






}  // namespace expt
}  // namespace internal
}  // namespace RAJA


#endif  // guard


#endif  // __RVVF__
