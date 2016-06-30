#ifndef _POLYBENCH_UTILS_HPP_
#define _POLYBENCH_UTILS_HPP_

// String conversion for argument parsing
namespace util
{
namespace convert
{

template <typename T>
T strTo(const char *)
{
  return T{};
}

template <>
int strTo<int>(const char *i)
{
  return std::stoi(i);
}
template <>
long strTo<long>(const char *i)
{
  return std::stol(i);
}
template <>
long long strTo<long long>(const char *i)
{
  return std::stoll(i);
}
template <>
unsigned long strTo<unsigned long>(const char *i)
{
  return std::stoul(i);
}
template <>
unsigned long long strTo<unsigned long long>(const char *i)
{
  return std::stoull(i);
}
template <>
float strTo<float>(const char *i)
{
  return std::stof(i);
}
template <>
double strTo<double>(const char *i)
{
  return std::stod(i);
}
template <>
long double strTo<long double>(const char *i)
{
  return std::stold(i);
}
}
}

// Repeat
namespace util
{

template <typename, typename>
struct append_to_type_seq {
};

template <typename T, typename... Ts, template <typename...> class TT>
struct append_to_type_seq<T, TT<Ts...>> {
  using type = TT<Ts..., T>;
};

template <typename T, unsigned int N, template <typename...> class TT>
struct repeat {
  using type =
    typename append_to_type_seq<T, typename repeat<T, N - 1, TT>::type>::type;
};

template <typename T, template <typename...> class TT>
struct repeat<T, 0, TT> {
  using type = TT<>;
};
}

// C++14 make_unique
namespace util
{
template <class T>
struct _Unique_if {
  typedef std::unique_ptr<T> _Single_object;
};

template <class T>
struct _Unique_if<T[]> {
  typedef std::unique_ptr<T[]> _Unknown_bound;
};

template <class T, size_t N>
struct _Unique_if<T[N]> {
  typedef void _Known_bound;
};

template <class T, class... Args>
typename _Unique_if<T>::_Single_object make_unique(Args &&... args)
{
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <class T>
typename _Unique_if<T>::_Unknown_bound make_unique(size_t n)
{
  typedef typename std::remove_extent<T>::type U;
  return std::unique_ptr<T>(new U[n]());
}

template <class T, class... Args>
typename _Unique_if<T>::_Known_bound make_unique(Args &&...) = delete;
}

// C++14 integer_sequence / index_sequence
namespace util
{
template <class T, T... Ints>
struct integer_sequence {
};

template <class S>
struct next_integer_sequence;

template <class T, T... Ints>
struct next_integer_sequence<integer_sequence<T, Ints...>> {
  using type = integer_sequence<T, Ints..., sizeof...(Ints)>;
};

template <class T, T I, T N>
struct make_int_seq_impl;

template <class T, T N>
using make_integer_sequence = typename make_int_seq_impl<T, 0, N>::type;

template <class T, T I, T N>
struct make_int_seq_impl {
  using type = typename next_integer_sequence<
    typename make_int_seq_impl<T, I + 1, N>::type>::type;
};

template <class T, T N>
struct make_int_seq_impl<T, N, N> {
  using type = integer_sequence<T>;
};

template <std::size_t... Ints>
using index_sequence = integer_sequence<std::size_t, Ints...>;

template <std::size_t N>
using make_index_sequence = make_integer_sequence<std::size_t, N>;
}

#endif
