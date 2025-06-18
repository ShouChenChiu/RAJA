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
#include <RAJA/policy/tensor/arch/rvv/traits.hpp>
#include <RAJA/policy/tensor/arch/rvv/rvv_int32.hpp>
#include <RAJA/policy/tensor/arch/rvv/rvv_int64.hpp>
#include <RAJA/policy/tensor/arch/rvv/rvv_float.hpp>
#include <RAJA/policy/tensor/arch/rvv/rvv_double.hpp>

#endif  // __RVVF__
