#ifndef _MULTIDIM_ARRAY_HPP_
#define _MULTIDIM_ARRAY_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <type_traits>
#include <utility>

#include <cstddef>

#include <AlignedMemory.hpp>

namespace util {

  template<class T> struct _Unique_if {
    typedef std::unique_ptr<T> _Single_object;
  };

  template<class T> struct _Unique_if<T[]> {
    typedef std::unique_ptr<T[]> _Unknown_bound;
  };

  template<class T, size_t N> struct _Unique_if<T[N]> {
    typedef void _Known_bound;
  };

  template<class T, class... Args>
  typename _Unique_if<T>::_Single_object
  make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
  }

  template<class T>
  typename _Unique_if<T>::_Unknown_bound
  make_unique(size_t n) {
    typedef typename std::remove_extent<T>::type U;
    return std::unique_ptr<T>(new U[n]());
  }

  template<class T, class... Args>
  typename _Unique_if<T>::_Known_bound
  make_unique(Args&&...) = delete;
}

template <typename T>
using Ptr = T* __restrict__;

template <typename T>
using CPtr = const Ptr<T>;

template <typename T>
using ManagedPtr = std::unique_ptr <T, mem::AlignedDeleter>;

template <typename T, size_t N>
class MultiDimArray {

  const std::array<size_t,N> extents;
  const std::array<size_t,N> coeffs;
  ManagedPtr<T> managedData;
  Ptr<T> rawData;

  inline Ptr<T> allocData () const noexcept {
    size_t allocSize { 1 };
    for (int i { 0 }; i < N; ++i)
      allocSize *= extents[i];
    return static_cast <Ptr<T>> (mem::defaultAllocator (allocSize * sizeof (T), 1024));
  }

  inline std::array<size_t,N> calculateCoeffs () const noexcept {
    std::array<size_t, N> res;
    size_t off { 1 };
    for (int i { N - 1 }; i >= 0; --i) {
      res [i] = off;
      off *= extents [i];
    }
    return res;
  }

  template <size_t Dim, typename Length, typename... Lengths>
  inline size_t computeOffset (Length curr, Lengths ... rest) const noexcept {
    return computeOffset<Dim+1> (rest ...) + extents[Dim] * curr;
  }

  template <size_t Dim, typename Length>
  inline size_t computeOffset (Length curr) const noexcept {
    return curr;
  }

public:

  MultiDimArray() = delete;

  template <typename... DimensionLengths>
  MultiDimArray<T,N> (DimensionLengths ... lengths) noexcept
    : extents {{static_cast<size_t>(lengths)...}},
      coeffs { calculateCoeffs () },
      managedData { allocData(), mem::defaultDeleter },
      rawData { managedData.get() } { }

  MultiDimArray<T,N> (const MultiDimArray<T,N> & rhs) noexcept
  : extents { rhs.extents },
    coeffs { rhs.coeffs },
    managedData { nullptr },
    rawData { rhs.rawData } { }

  MultiDimArray<T,N> (MultiDimArray<T,N> && rhs) noexcept
  : extents { std::move(rhs.extents) },
    coeffs { std::move(rhs.coeffs) },
    managedData { nullptr },
    rawData { rhs.rawData } { }

  inline void swap (MultiDimArray<T,N> &rhs) noexcept {
    using std::swap;
    swap (extents, rhs.extents);
    swap (coeffs, rhs.coeffs);
    swap (rawData, rhs.rawData);
  }

  inline MultiDimArray<T,N>& operator= (MultiDimArray<T,N> rhs) noexcept {
    this->swap (rhs);
    rhs.clear();
    return *this;
  }

  template <typename... Ind>
  inline T& operator()(Ind... indices) noexcept {
    return rawData [computeOffset<0>(indices...)];
  }

  template <typename... Ind>
  inline T& at(Ind... indices) noexcept {
    return rawData [computeOffset<0>(indices...)];
  }

  template <typename... Ind>
  inline const T& operator()(Ind... indices) const noexcept {
    return rawData [computeOffset<0>(indices...)];
  }

  template <typename... Ind>
  inline const T& at(Ind... indices) const noexcept {
    return rawData [computeOffset<0>(indices...)];
  }

};

template<typename T, size_t N>
inline void swap (MultiDimArray<T,N> & a, MultiDimArray<T,N> & b) noexcept {
  a.swap (b);
}

template<typename T> using Arr1D = MultiDimArray<T,1>;
template<typename T> using Arr2D = MultiDimArray<T,2>;
template<typename T> using Arr3D = MultiDimArray<T,3>;
template<typename T> using Arr4D = MultiDimArray<T,4>;

#endif
