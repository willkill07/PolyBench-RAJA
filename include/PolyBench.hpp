#ifndef POLYBENCH_HPP
#define POLYBENCH_HPP

#ifndef POLYBENCH_CACHE_SIZE_KB
#define POLYBENCH_CACHE_SIZE_KB 32770
#endif

#ifndef POLYBENCH_CACHE_LINE_SIZE_B
#define POLYBENCH_CACHE_LINE_SIZE_B 64
#endif

#include <numeric>
#include <string>
#include <vector>

#include <RAJA/RAJA.hxx>

#include "MultiDimArray.hpp"
#include "PolyBenchKernel.hpp"

#include "PolyBenchUtils.hpp"

template <typename T>
using CountArgs = typename std::tuple_element<0, T>::type::arg_count;
template <typename T>
using GetArgs = typename std::tuple_element<0, T>::type::args;

using dummy_t = double;

namespace detail {
template <typename TT, typename Arr, std::size_t... I>
TT parseArgs(Arr &&argv, util::index_sequence<I...>) {
  return std::make_tuple(
    ::util::convert::strTo<typename std::tuple_element<I, TT>::type>(
      argv[I])...);
}
}

template <typename TT,
          typename Arr,
          typename Ind = util::make_index_sequence<std::tuple_size<TT>::value>>
TT parseArgs(Arr &&argv) {
  return detail::parseArgs<TT>(argv, Ind{});
}

class KernelPacker {
  std::vector<std::unique_ptr<PolyBenchKernel>> kernels;

  template <typename T, size_t... I, typename... A>
  std::unique_ptr<T> create(std::tuple<A...> args, util::index_sequence<I...>) {
    // We need to invoke the appropriate constructor
    // Steps:
    //  - expand parameter pack through std::get<I>(args)...
    //  - forward each type because make_unique expects l-values
    //  - in order to get the correct type for forwarding,
    //    typename std::tuple_element<I,TT>::type must be used
    //
    // Ideally, this should be split on multiple lines, but the
    // parameter pack expansion dependent type 'I' is used in all parts
    using TT = std::tuple<A...>;
    return util::make_unique<T, A...>(
      std::forward<typename std::tuple_element<I, TT>::type>(
        std::get<I>(args))...);
  }

  template <std::size_t I = 0, typename TupleType, typename Tp>
  inline typename std::enable_if<I == std::tuple_size<Tp>::value, void>::type
  add_impl(TupleType &&args) {
  }

  template <std::size_t I = 0, typename TupleType, typename Tp, typename = typename std::enable_if< I < std::tuple_size<Tp>::value>::type>
    inline void add_impl(TupleType &&args) {
    // Type of current execution
    using Type = typename std::tuple_element<I, Tp>::type;
    // Type of args
    using TT = typename std::decay<TupleType>::type;
    // Number of args
    constexpr size_t Size = std::tuple_size<TT>::value;
    // Creation of new kernel
    kernels.push_back(create<Type>(args, util::make_index_sequence<Size>{}));
    add_impl<I + 1, TupleType, Tp>(args);
  }

public:
  template <typename V, typename TupleType>
  void addAll(TupleType &&args) {
    add_impl<0, TupleType, V>(args);
  }

  template <typename T, typename TupleType>
  void add(TupleType &&args) {
    addAll<std::tuple<T>>(args);
  }

  inline void run() {
    for (auto &k : kernels)
      k->run();
  }

  inline bool check() {
    return !std::accumulate(
      std::begin(kernels) + 1,
      std::end(kernels),
      true,
      [this](bool result, const std::unique_ptr<PolyBenchKernel> &k) {
        return this->kernels[0]->compareTo(k.get()) && result;
      });
  }
};

#endif /* !POLYBENCH_RAJA_HPP */
