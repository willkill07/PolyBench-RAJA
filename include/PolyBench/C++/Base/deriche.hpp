#ifndef _CPP_BASE_DERICHE_HPP_
#define _CPP_BASE_DERICHE_HPP_

#include "PolyBench/Base/deriche.hpp"

namespace CPlusPlus
{
namespace Base
{
template <typename T>
class deriche : public ::Base::deriche<T>
{
  using Parent = ::Base::deriche<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  deriche(Args... args)
  : ::Base::deriche<T>{std::string{"DERICHE - C++ Base"}, args...}
  {
  }

  virtual void init()
  {
    USE(READ, w, h);
    USE(READWRITE, imgIn);
    for (int i = 0; i < w; i++)
      for (int j = 0; j < h; j++)
        imgIn->at(i, j) =
          static_cast<T>((313 * i + 991 * j) % 65536) / static_cast<T>(65535.0);
  }

  virtual void exec()
  {
    USE(READ, alpha, w, h, imgIn);
    USE(READWRITE, imgOut, y1, y2);
    T xm1, tm1, ym1, ym2;
    T xp1, xp2;
    T tp1, tp2;
    T yp1, yp2;
    T k;
    T a1, a2, a3, a4, a5, a6, a7, a8;
    T b1, b2, c1, c2;
    k = (static_cast<T>(1.0) - exp(-alpha))
        * (static_cast<T>(1.0) - exp(-alpha))
        / (static_cast<T>(1.0) + static_cast<T>(2.0) * alpha * exp(-alpha)
           - exp(static_cast<T>(2.0) * alpha));
    a1 = a5 = k;
    a2 = a6 = k * exp(-alpha) * (alpha - static_cast<T>(1.0));
    a3 = a7 = k * exp(-alpha) * (alpha + static_cast<T>(1.0));
    a4 = a8 = -k * exp(static_cast<T>(-2.0) * alpha);
    b1 = pow(static_cast<T>(2.0), -alpha);
    b2 = -exp(static_cast<T>(-2.0) * alpha);
    c1 = c2 = 1;
    for (int i = 0; i < w; i++) {
      ym1 = static_cast<T>(0.0);
      ym2 = static_cast<T>(0.0);
      xm1 = static_cast<T>(0.0);
      for (int j = 0; j < h; j++) {
        y1->at(i, j) = a1 * imgIn->at(i, j) + a2 * xm1 + b1 * ym1 + b2 * ym2;
        xm1 = imgIn->at(i, j);
        ym2 = ym1;
        ym1 = y1->at(i, j);
      }
    }
    for (int i = 0; i < w; i++) {
      yp1 = static_cast<T>(0.0);
      yp2 = static_cast<T>(0.0);
      xp1 = static_cast<T>(0.0);
      xp2 = static_cast<T>(0.0);
      for (int j = h - 1; j >= 0; j--) {
        y2->at(i, j) = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
        xp2 = xp1;
        xp1 = imgIn->at(i, j);
        yp2 = yp1;
        yp1 = y2->at(i, j);
      }
    }
    for (int i = 0; i < w; i++)
      for (int j = 0; j < h; j++) {
        imgOut->at(i, j) = c1 * (y1->at(i, j) + y2->at(i, j));
      }
    for (int j = 0; j < h; j++) {
      tm1 = static_cast<T>(0.0);
      ym1 = static_cast<T>(0.0);
      ym2 = static_cast<T>(0.0);
      for (int i = 0; i < w; i++) {
        y1->at(i, j) = a5 * imgOut->at(i, j) + a6 * tm1 + b1 * ym1 + b2 * ym2;
        tm1 = imgOut->at(i, j);
        ym2 = ym1;
        ym1 = y1->at(i, j);
      }
    }
    for (int j = 0; j < h; j++) {
      tp1 = static_cast<T>(0.0);
      tp2 = static_cast<T>(0.0);
      yp1 = static_cast<T>(0.0);
      yp2 = static_cast<T>(0.0);
      for (int i = w - 1; i >= 0; i--) {
        y2->at(i, j) = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
        tp2 = tp1;
        tp1 = imgOut->at(i, j);
        yp2 = yp1;
        yp1 = y2->at(i, j);
      }
    }
    for (int i = 0; i < w; i++)
      for (int j = 0; j < h; j++)
        imgOut->at(i, j) = c2 * (y1->at(i, j) + y2->at(i, j));
  }
};
} // Base
} // CPlusPlus
#endif
