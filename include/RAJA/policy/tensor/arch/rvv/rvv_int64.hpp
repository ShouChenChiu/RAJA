/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining a SIMD register abstraction.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// #include <cstdint>
#ifdef __RVVF__
#ifndef RAJA_policy_vector_register_rvv_int64_t_HPP
#define RAJA_policy_vector_register_rvv_int64_t_HPP

#define VTYPE_IMPL(dtype, MUL) v##dtype##m##MUL##_t
#define VSETMAX(LANE, __LMUL__) __riscv_vsetvlmax_e##LANE##m##__LMUL__
#define OP_IMPL(NAME, TYPE, DTYPE, MUL, ATTR) __riscv_##NAME##_##TYPE##_##DTYPE##m##MUL##ATTR
#define REINTERPRE_IMPL(SRC_TYPE, DES_TYPE, MUL) __riscv_vreinterpret_v_##SRC_TYPE##m##MUL##_##DES_TYPE##m##MUL

#define VTYPE(type, MUL) VTYPE_IMPL(type, MUL)
#define VSET(LANE, __LMUL__) VSETMAX(LANE, __LMUL__)
#define VOP(NAME, TYPE, DTYPE, MUL, ATTR) OP_IMPL(NAME, TYPE, DTYPE, MUL, ATTR)
#define VREINTERPRET(SRC_TYPE, DES_TYPE, MUL) REINTERPRE_IMPL(SRC_TYPE, DES_TYPE, MUL)


#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/pattern/tensor/internal/RegisterBase.hpp"

// Include SIMD intrinsics header file
#include <riscv_vector.h>
#include <cmath>

namespace RAJA
{
namespace expt
{


#ifndef __VL__
#define __VL__ 256
#endif 

#ifndef __LMUL__
#define __LMUL__ 1
#endif




typedef VTYPE(int64, __LMUL__) fixed_int64_t __attribute__((riscv_rvv_vector_bits(__VL__ * __LMUL__)));

template<>
class Register<int64_t, rvv_register>
    : public internal::expt::RegisterBase<Register<int64_t, rvv_register>>
{
public:
  using base_type =
      internal::expt::RegisterBase<Register<int64_t, rvv_register>>;

  using register_policy = rvv_register;
  using self_type       = Register<int64_t, rvv_register>;
  using element_type    = int64_t;
  using register_type   = fixed_int64_t;
  using int_vector_type = Register<int64_t, rvv_register>;

private:
  register_type m_value;
  size_t vl;

  RAJA_INLINE
  auto createMask(camp::idx_t N) const {
    if constexpr (__LMUL__ == 1){
      auto idx = VOP(vid, v, u64, 1,)(vl);
      return VOP(vmsltu, vx, u64, 1, _b64)(idx, (uint64_t)N, vl);
    }
    else if constexpr(__LMUL__ == 2){
      auto idx = VOP(vid, v, u64, 2,)(vl);
      return VOP(vmsltu, vx, u64, 2, _b32)(idx, (uint64_t)N, vl);
    }
    else if constexpr(__LMUL__ == 4){
      auto idx = VOP(vid, v, u64, 4,)(vl);
      return VOP(vmsltu, vx, u64, 4, _b16)(idx, (uint64_t)N, vl);
    }
    else if constexpr(__LMUL__ == 8){
      auto idx = VOP(vid, v, u64, 8,)(vl);
      return VOP(vmsltu, vx, u64, 8, _b8)(idx, (uint64_t)N, vl);
    }   
  }  
public:
  static constexpr camp::idx_t s_num_elem = __VL__ * __LMUL__ / 64;

  /*!
   * @brief Default constructor, zeros register contents
   */
  RAJA_INLINE
  Register(){
    vl = VSET(64, __LMUL__)();
    m_value = VOP(vmv, v_x, i64, __LMUL__, )((int64_t)0, vl);
  }

  /*!
   * @brief Copy constructor from underlying simd register
   */
  RAJA_INLINE
  explicit Register(register_type const& c) : m_value(c), vl(VSET(64, __LMUL__)()) {}

  /*!
   * @brief Copy constructor
   */
  RAJA_INLINE
  Register(self_type const& c) : base_type(c), m_value(c.m_value), vl(VSET(64, __LMUL__)()) {}

  /*!
   * @brief Copy assignment constructor
   */
  RAJA_INLINE
  self_type& operator=(self_type const& c)
  {
    m_value = c.m_value;
    vl = c.vl;
    return *this;
  }

  /*!
   * @brief Construct from scalar.
   * Sets all elements to same value (broadcast).
   */
  RAJA_INLINE
  Register(element_type const& c) : vl(VSET(64, __LMUL__)()) {
    m_value = VOP(vmv, v_x, i64, __LMUL__, )((int64_t)c, vl);
  }

  /*!
   * @brief Returns underlying SIMD register.
   */
  RAJA_INLINE
  constexpr register_type get_register() const { return m_value; }

  /*!
   * @brief Load a full register from a stride-one memory location
   *
   */
  RAJA_INLINE
  self_type& load_packed(element_type const* ptr)
  {
#ifdef RAJA_ENABLE_VECTOR_STATS
    RAJA::tensor_stats::num_vector_load_packed++;
#endif
    m_value = VOP(vle64, v, i64, __LMUL__, )(ptr,  vl);
    return *this;
  }

  /*!
   * @brief Partially load a register from a stride-one memory location given
   *        a run-time number of elements.
   *
   */
  RAJA_INLINE
  self_type& load_packed_n(element_type const* ptr, camp::idx_t N)
  {
#ifdef RAJA_ENABLE_VECTOR_STATS
    RAJA::tensor_stats::num_vector_load_packed_n++;
#endif
    m_value = VOP(vle64, v, i64, __LMUL__, _m)(createMask(N), ptr, vl);
    return *this;
  }

  /*!
   * @brief Gather a full register from a strided memory location
   *
   */
  RAJA_INLINE
  self_type& load_strided(element_type const* ptr, camp::idx_t stride)
  {
#ifdef RAJA_ENABLE_VECTOR_STATS
    RAJA::tensor_stats::num_vector_load_strided++;
#endif
    m_value = VOP(vlse64, v, i64, __LMUL__, )(ptr, stride * sizeof(element_type), vl);
    return *this;
  }

  /*!
   * @brief Partially load a register from a stride-one memory location given
   *        a run-time number of elements.
   *
   */
  RAJA_INLINE
  self_type& load_strided_n(element_type const* ptr,
                            camp::idx_t stride,
                            camp::idx_t N)
  {
#ifdef RAJA_ENABLE_VECTOR_STATS
    RAJA::tensor_stats::num_vector_load_strided_n++;
#endif
    ptrdiff_t byte_stride = (ptrdiff_t)(stride * sizeof(int64_t));
    m_value = VOP(vlse64, v, i64, __LMUL__, _m)(createMask(N), ptr, byte_stride, vl);
    return *this;
  }

/*!
 * @brief Generic gather operation for full vector.
 *
 * Must provide another register containing offsets of all values
 * to be loaded relative to supplied pointer.
 *
 * Offsets are element-wise, not byte-wise.
 *
 */
  RAJA_INLINE
  self_type& gather(element_type const* ptr, int_vector_type offsets)
  {
#ifdef RAJA_ENABLE_VECTOR_STATS
    RAJA::tensor_stats::num_vector_load_strided_n++;
#endif
    auto index = VREINTERPRET(i64, u64, __LMUL__)(offsets.get_register());
    m_value = VOP(vluxei64, v, i64, __LMUL__, )(ptr, index, vl);
    return *this;
  }

/*!
 * @brief Generic gather operation for n-length subvector.
 *
 * Must provide another register containing offsets of all values
 * to be loaded relative to supplied pointer.
 *
 * Offsets are element-wise, not byte-wise.
 *
 */
  RAJA_INLINE
  self_type& gather_n(element_type const* ptr,
                      int_vector_type offsets,
                      camp::idx_t N)
  {
#ifdef RAJA_ENABLE_VECTOR_STATS
    RAJA::tensor_stats::num_vector_load_strided_n++;
#endif
    auto index = VREINTERPRET(i64, u64, __LMUL__)(offsets.get_register());
    m_value = VOP(vluxei64, v, i64, __LMUL__, _m)(createMask(N), ptr, index, vl);
    return *this;
  }

  /*!
   * @brief Store entire register to consecutive memory locations
   *
   */
  RAJA_INLINE
  self_type const& store_packed(element_type* ptr) const
  {
#ifdef RAJA_ENABLE_VECTOR_STATS
    RAJA::tensor_stats::num_vector_store_packed++;
#endif
    //__riscv_vse64_v_i64m1(ptr, m_value, vl);
    VOP(vse64, v, i64, __LMUL__, )(ptr, m_value, vl);
    return *this;
  }

  /*!
   * @brief Store entire register to consecutive memory locations
   *
   */
  RAJA_INLINE
  self_type const& store_packed_n(element_type* ptr, camp::idx_t N) const
  {
#ifdef RAJA_ENABLE_VECTOR_STATS
    RAJA::tensor_stats::num_vector_store_packed_n++;
#endif
    VOP(vse64, v, i64, __LMUL__, _m)(createMask(N), ptr, m_value, vl);
    return *this;
  }

  /*!
   * @brief Store entire register to consecutive memory locations
   *
   */
  RAJA_INLINE
  self_type const& store_strided(element_type* ptr, camp::idx_t stride) const
  {
#ifdef RAJA_ENABLE_VECTOR_STATS
    RAJA::tensor_stats::num_vector_store_strided++;
#endif
    VOP(vsse64, v, i64, __LMUL__, )(ptr, stride * sizeof(element_type), m_value, vl);
    return *this;
  }

  /*!
   * @brief Store partial register to consecutive memory locations
   *
   */
  RAJA_INLINE
  self_type const& store_strided_n(element_type* ptr,
                                   camp::idx_t stride,
                                   camp::idx_t N) const
  {
#ifdef RAJA_ENABLE_VECTOR_STATS
    RAJA::tensor_stats::num_vector_store_strided_n++;
#endif
    VOP(vsse64, v, i64, __LMUL__, _m)(createMask(N), ptr, stride * sizeof(element_type), m_value, vl);
    return *this;
  }

  /*!
   * @brief Get scalar value from vector register
   * @param i Offset of scalar to get
   * @return Returns scalar value at i
   */
  RAJA_INLINE
  element_type get(camp::idx_t i) const { 
    element_type temp[vl];
    VOP(vse64, v, i64, __LMUL__,)(temp, m_value, vl);
    return temp[i];
   }

  /*!
   * @brief Set scalar value in vector register
   * @param i Offset of scalar to set
   * @param value Value of scalar to set
   */
  RAJA_INLINE
  self_type& set(element_type value, camp::idx_t i)
  {
    element_type temp[vl];
    VOP(vse64, v, i64, __LMUL__, )(temp, m_value, vl);
    temp[i] = value;
    m_value = VOP(vle64, v, i64, __LMUL__, )(temp, vl);
    return *this;
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& broadcast(element_type const& value)
  {
    m_value = VOP(vmv, v_x, i64, __LMUL__, )(value, vl);
    return *this;
  }

  /*!
   * @brief Extracts a scalar value and broadcasts to a new register
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type get_and_broadcast(int i) const
  {
    if (i >= vl)
      return *this;
    element_type temp[vl];
    VOP(vse64, v, i64, __LMUL__, )(temp, m_value, vl);
    element_type elem = temp[i];
    self_type result;
    result.m_value = VOP(vmv, v_x, i64, __LMUL__, )(elem, vl);
    return result;
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& copy(self_type const& src)
  {
    m_value = src.m_value;
    return *this;
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type add(self_type const& b) const
  {
    return self_type(VOP(vadd, vv, i64, __LMUL__, )(m_value, b.m_value, vl));
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type subtract(self_type const& b) const
  {
    return self_type(VOP(vsub, vv, i64, __LMUL__, )(m_value, b.m_value, vl));
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type multiply(self_type const& b) const
  {
    return self_type(VOP(vmul, vv, i64, __LMUL__, )(m_value, b.m_value, vl));
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type divide(self_type const& b) const
  {
    return self_type(VOP(vdiv, vv, i64, __LMUL__, )(m_value, b.m_value, vl));
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type divide_n(self_type const& b, camp::idx_t N) const
  {
    return self_type(VOP(vdiv, vv, i64, __LMUL__, _m)(createMask(N), m_value, b.m_value, vl));
  }

// only use FMA's if the compiler has them turned on
#ifdef __FMA__
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type multiply_add(self_type const& b, self_type const& c) const
  {
    return self_type(VOP(vmadd, vv, i64, __LMUL__,)(c.m_value, m_value, b.m_value, vl));
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type multiply_subtract(self_type const& b, self_type const& c) const
  {
    return self_type(VOP(vmsub, vv, i64, __LMUL__,)(c.m_value, m_value, b.m_value, vl));
  }
#endif

//   /*!
//    * @brief Sum the elements of this vector
//    * @return Sum of the values of the vectors scalar elements
//    */
  RAJA_INLINE
  element_type sum(camp::idx_t N = __LMUL__ * 4) const
  {
    auto init_scale = VOP(vmv, v_x, i64, 1,)((int64_t)0, vl);
    auto vsum = VOP(vredsum, vs, i64, __LMUL__, _i64m1_m)(createMask(N), m_value, init_scale, vl); 
    element_type result[4];
    VOP(vse64, v, i64, 1,)(result, vsum, vl);
    return result[0];
  }

  /*!
   * @brief Returns the largest element
   * @return The largest scalar element in the register
   */
  RAJA_INLINE
  element_type max(camp::idx_t N = __LMUL__ * 4) const
  {
    auto e_ = VOP(vmv, v_x, i64, 1,)(-INFINITY, 1);
    auto vmax = VOP(vredmax, vs, i64, __LMUL__, _i64m1_m)(createMask(N), m_value, e_, vl);
    element_type result[4];
    VOP(vse64, v, i64, 1,)(result, vmax, vl);
    return result[0];
  }

  /*!
   * @brief Returns element-wise largest values
   * @return Vector of the element-wise max values
   */
  RAJA_INLINE
  self_type vmax(self_type a) const
  {
    return self_type(VOP(vmax, vv, i64, __LMUL__, )(m_value, a.m_value, vl));
  }

  /*!
   * @brief Returns the largest element
   * @return The largest scalar element in the register
   */
  RAJA_INLINE
  element_type min() const
  {
    auto e_ = VOP(vmv, v_x, i64, 1,)(INFINITY, 1);
    auto vmin = VOP(vredmin, vs, i64, __LMUL__, _i64m1)(m_value, e_, vl);
    element_type result[4];
    VOP(vse64, v, i64, 1,)(result, vmin, vl);
    return result[0];
  }

  /*!
   * @brief Returns the largest element from first N lanes
   * @return The largest scalar element in the register
   */
  RAJA_INLINE
  element_type min_n(camp::idx_t N) const
  {
    auto e_ = VOP(vmv, v_x, i64, 1,)(INFINITY, 1);
    auto vmin_n = VOP(vredmin, vs, i64, __LMUL__, _i64m1_m)(createMask(N), m_value, e_, vl);
    element_type result[4];
    VOP(vse64, v, i64, 1,)(result, vmin_n, vl);
    return result[0];
  }

  /*!
   * @brief Returns element-wise largest values
   * @return Vector of the element-wise max values
   */
  RAJA_INLINE
  self_type vmin(self_type a) const
  {
    return self_type(VOP(vmin, vv, i64, __LMUL__, )(m_value, a.m_value, vl));
  }
// 
};


}  // namespace expt

}  // namespace RAJA


#endif

#endif  //__RVVF__
