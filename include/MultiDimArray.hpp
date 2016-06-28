#ifndef _MULTIDIM_ARRAY_HPP_
#define _MULTIDIM_ARRAY_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <type_traits>
#include <utility>

#include <cmath>
#include <cstddef>

#include <iostream>

template <typename T>
using Ptr = T *__restrict__;

template <typename T>
using CPtr = const Ptr<T>;

constexpr const int MULTIDIMARRAY_ALIGNMENT = 512;

template <typename T, size_t N>
class MultiDimArray {
  using DimArray = std::array<size_t, N>;

  DimArray extents{};
  DimArray coeffs{};
  size_t size{0};
  Ptr<T> rawData{nullptr};
  bool owner{false};

  inline size_t calculateSize() const noexcept {
    size_t allocSize{1};
    for (size_t i{0}; i < N; ++i)
      allocSize *= extents[i];
    return allocSize;
  }

  inline Ptr<T> allocData() const noexcept {
    void *data;
    if (posix_memalign(&data, MULTIDIMARRAY_ALIGNMENT, size * sizeof(T))) {
      exit(-1);
    }
    return static_cast<Ptr<T>>(data);
  }

  inline DimArray calculateCoeffs() const noexcept {
    DimArray res;
    size_t off{1};
    for (auto i = int{N - 1}; i >= 0; --i) {
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
  using iterator = T *;
  using const_iterator = const iterator;
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = T &;
  using const_reference = const T &;

  iterator begin() {
    return rawData;
  }
  const_iterator begin() const {
    return rawData;
  }
  const_iterator cbegin() const {
    return rawData;
  }
  iterator end() {
    return rawData + size;
  }
  const_iterator end() const {
    return rawData + size;
  }
  const_iterator cend() const {
    return rawData + size;
  }

  MultiDimArray() = delete;

  DimArray dims() {
    return extents;
  }
  DimArray offsets() {
    return coeffs;
  }

  template <size_t M>
  typename std::enable_if<(M < N), size_t>::type dim() {
    return extents[M];
  }

  template <
    typename... DimensionLengths,
    typename = typename std::enable_if<sizeof...(DimensionLengths) == N>::type>
  explicit MultiDimArray<T, N>(DimensionLengths... lengths) noexcept {
    extents = DimArray{{static_cast<size_t>(lengths)...}};
    coeffs = calculateCoeffs();
    size = calculateSize();
    rawData = allocData();
    owner = true;
  }

  ~MultiDimArray<T, N>() {
    if (owner) {
      free(rawData);
      rawData = nullptr;
      owner = false;
    }
  }

  inline void swap(MultiDimArray<T, N> &rhs) noexcept {
    if (&rhs != this) {
      using std::swap;
      swap(extents, rhs.extents);
      swap(coeffs, rhs.coeffs);
      swap(size, rhs.size);
      swap(owner, rhs.owner);
      swap(rawData, rhs.rawData);
    }
  }

  inline MultiDimArray<T, N> &operator=(MultiDimArray<T, N> rhs) noexcept {
    this->swap(rhs);
    return *this;
  }

  template <typename... Ind>
  inline typename std::enable_if<sizeof...(Ind) == N, T &>::type operator()(
    Ind... indices) noexcept {
    Ptr<T> ptr = static_cast<Ptr<T>>(
      __builtin_assume_aligned(rawData, MULTIDIMARRAY_ALIGNMENT));
    return ptr[computeOffset<0>(indices...)];
  }

  template <typename... Ind>
  inline typename std::enable_if<sizeof...(Ind) == N, T &>::type at(
    Ind... indices) noexcept {
    Ptr<T> ptr = static_cast<Ptr<T>>(
      __builtin_assume_aligned(rawData, MULTIDIMARRAY_ALIGNMENT));
    return ptr[computeOffset<0>(indices...)];
  }

  template <typename... Ind>
  inline typename std::enable_if<sizeof...(Ind) == N, const T &>::type
  operator()(Ind... indices) const noexcept {
    CPtr<T> ptr = static_cast<Ptr<T>>(
      __builtin_assume_aligned(rawData, MULTIDIMARRAY_ALIGNMENT));
    return ptr[computeOffset<0>(indices...)];
  }

  template <typename... Ind>
  inline typename std::enable_if<sizeof...(Ind) == N, const T &>::type at(
    Ind... indices) const noexcept {
    CPtr<T> ptr = static_cast<Ptr<T>>(
      __builtin_assume_aligned(rawData, MULTIDIMARRAY_ALIGNMENT));
    return ptr[computeOffset<0>(indices...)];
  }

  static bool compare(
    const MultiDimArray<T, N> *a,
    const MultiDimArray<T, N> *b,
    T epsilon,
    bool print = true) {
    auto res =
      std::mismatch(a->begin(), a->end(), b->begin(), [epsilon](T va, T vb) {
        return std::abs(va - vb) <= epsilon;
      });
    if (print && res.first != a->end()) {
      std::cerr << "mismatch occured at index " << (res.first - a->begin())
                << '\n'
                << "A[i] = " << *(res.first) << " -- B[i] = " << *(res.second)
                << std::endl;
    }
    return (res.first == a->end());
  }
};

template <typename T, size_t N>
inline void swap(MultiDimArray<T, N> &a, MultiDimArray<T, N> &b) noexcept {
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
