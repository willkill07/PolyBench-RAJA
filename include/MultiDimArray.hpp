#ifndef _MULTIDIM_ARRAY_HPP_
#define _MULTIDIM_ARRAY_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <type_traits>
#include <utility>

#include <cstddef>
#include <cmath>

template <typename T>
using Ptr = T* __restrict__;

template <typename T>
using CPtr = const Ptr<T>;

template <typename T, size_t N>
class MultiDimArray {
  using DimArray = std::array<size_t, N>;

  DimArray extents {};
  DimArray coeffs {};
	size_t size { 0 };
  Ptr<T> rawData { nullptr };
  bool owner { false };

  inline size_t calculateSize() const noexcept {
    size_t allocSize{1};
    for (int i{0}; i < N; ++i)
      allocSize *= extents[i];
		return allocSize;
  }

  inline Ptr<T> allocData() const noexcept {
    void* data;
    posix_memalign(&data, 1024, size * sizeof (T));
    return static_cast<Ptr<T>>(data);
  }

  inline DimArray calculateCoeffs() const noexcept {
    DimArray res;
    size_t off{1};
    for (int i{N - 1}; i >= 0; --i) {
      res[i] = off;
      off *= extents[i];
    }
    return res;
  }

  template <size_t Dim, typename Length, typename... Lengths>
  inline size_t computeOffset(Length next, Lengths... rest) const noexcept {
    return next * coeffs[Dim] + computeOffset<Dim + 1>(rest...);
  }

  template <size_t Dim, typename Length>
  inline size_t computeOffset(Length next) const noexcept {
    return next;
  }

 public:
	using iterator = T*;
	using const_iterator = const iterator;
	using value_type = T;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
	using reference = T&;
	using const_reference = const T&;

	iterator begin() { return rawData; }
	const_iterator begin() const { return rawData; }
	const_iterator cbegin() const { return rawData; }
	iterator end() { return rawData + size; }
	const_iterator end() const { return rawData + size; }
	const_iterator cend() const { return rawData + size; }

  MultiDimArray() = delete;

  DimArray dims() { return extents; }
  DimArray offsets() { return coeffs; }

  template <size_t M>
  typename std::enable_if<(M<N),size_t>::type
  dim() {
    return extents[M];
  }

  template <typename... DimensionLengths>
  explicit MultiDimArray<T, N>(DimensionLengths... lengths) noexcept {
    extents = DimArray { static_cast<size_t>(lengths)... };
    coeffs = calculateCoeffs();
    size = calculateSize();
    rawData = allocData();
    owner = true;
  }

  ~MultiDimArray<T, N>() {
    if (owner) {
      free (rawData);
      rawData = nullptr;
      owner = false;
    }
  }

  inline void swap(MultiDimArray<T, N>& rhs) noexcept {
    if (&rhs != this) {
      using std::swap;
      swap(extents, rhs.extents);
      swap(coeffs, rhs.coeffs);
      swap(size, rhs.size);
      swap(owner, rhs.owner);
      swap(rawData, rhs.rawData);
    }
  }

  inline MultiDimArray<T, N>& operator=(MultiDimArray<T, N> rhs) noexcept {
    this->swap(rhs);
    return *this;
  }

  template <typename... Ind>
  inline T& operator()(Ind... indices) noexcept {
    return rawData[computeOffset<0>(indices...)];
  }

  template <typename... Ind>
  inline T& at(Ind... indices) noexcept {
    return rawData[computeOffset<0>(indices...)];
  }

  template <typename... Ind>
  inline const T& operator()(Ind... indices) const noexcept {
    return rawData[computeOffset<0>(indices...)];
  }

  template <typename... Ind>
  inline const T& at(Ind... indices) const noexcept {
    return rawData[computeOffset<0>(indices...)];
  }

	static bool compare (const MultiDimArray<T,N>* a, const MultiDimArray<T,N>* b, T epsilon) {
    auto res = std::mismatch (a->begin(), a->end(), b->begin(), [epsilon] (T va, T vb) {
        return std::abs (va - vb) <= epsilon;
      });
    return (res.first == a->end());
  }

};

template <typename T, size_t N>
inline void swap(MultiDimArray<T, N>& a, MultiDimArray<T, N>& b) noexcept {
  a.swap(b);
}

template <typename T>
using Arr1D = MultiDimArray<T, 1>;
template <typename T>
using Arr2D = MultiDimArray<T, 2>;
template <typename T>
using Arr3D = MultiDimArray<T, 3>;
template <typename T>
using Arr4D = MultiDimArray<T, 4>;

#endif
