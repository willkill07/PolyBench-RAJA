#ifndef _RAJA_BASE_DERICHE_HPP_
#define _RAJA_BASE_DERICHE_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/deriche.hpp"

namespace RAJA {
namespace Base {
template <typename T>
class deriche : public ::Base::deriche<T> {
  using Parent = ::Base::deriche<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  deriche(Args... args)
      : ::Base::deriche<T>{std::string{"DERICHE - RAJA Base"}, args...} {
  }

  virtual void init() {
    USE(READ, w, h);
    USE(READWRITE, imgIn);
    using init_pol = NestedPolicy<ExecList<simd_exec, simd_exec>>;
    forallN<init_pol>(
      RangeSegment{0, w},
      RangeSegment{0, h},
      [=](int i, int j) {
        imgIn->at(i, j) =
          static_cast<T>((313 * i + 991 * j) % 65536) / static_cast<T>(65535);
      });
  }

  virtual void exec() {
    USE(READ, w, h, imgIn, alpha);
    T k = (1.0 - exp(-alpha)) * (1.0 - exp(-alpha))
          / (1.0 + 2.0 * alpha * exp(-alpha) - exp(2.0 * alpha));
    T a1 = k;
    T a2 = k * exp(-alpha) * (alpha - 1.0);
    T a3 = k * exp(-alpha) * (alpha + 1.0);
    T a4 = -k * exp(-2.0 * alpha);
    T a5 = k;
    T a6 = k * exp(-alpha) * (alpha - 1.0);
    T a7 = k * exp(-alpha) * (alpha + 1.0);
    T a8 = -k * exp(-2.0 * alpha);
    T b1 = pow(2.0, -alpha);
    T b2 = -exp(-2.0 * alpha);
    T c1 = 1;
    T c2 = 1;
    using exec_pol = NestedPolicy<ExecList<simd_exec, simd_exec>>;
    {
      USE(READWRITE, y1);
      forall<simd_exec>(0, w, [=](int i) {
        T ym1{0.0}, ym2{0.0}, xm1{0.0};
        for (int j = 0; j < h; ++j) {
          y1->at(i, j) = a1 * imgIn->at(i, j) + a2 * xm1 + b1 * ym1 + b2 * ym2;
          xm1 = imgIn->at(i, j);
          ym2 = ym1;
          ym1 = y1->at(i, j);
        }
      });
    }
    {
      USE(READWRITE, y2);
      forall<simd_exec>(0, w, [=](int i) {
        T yp1{0.0}, yp2{0.0}, xp1{0.0}, xp2{0.0};
        for (int j = h - 1; j >= 0; --j) {
          y2->at(i, j) = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
          xp2 = xp1;
          xp1 = imgIn->at(i, j);
          yp2 = yp1;
          yp1 = y2->at(i, j);
        }
      });
    }
    {
      USE(READ, y2, y1);
      USE(READWRITE, imgOut);
      forallN<exec_pol>(
        RangeSegment{0, w},
        RangeSegment{0, h},
        [=](int i, int j) {
          imgOut->at(i, j) = c1 * (y1->at(i, j) + y2->at(i, j));
        });
    }
    {
      USE(READ, imgOut);
      USE(READWRITE, y1);
      forall<simd_exec>(0, h, [=](int j) {
        T ym1{0.0}, ym2{0.0}, tm1{0.0};
        for (int i = 0; i < w; ++i) {
          y1->at(i, j) = a5 * imgOut->at(i, j) + a6 * tm1 + b1 * ym1 + b2 * ym2;
          tm1 = imgOut->at(i, j);
          ym2 = ym1;
          ym1 = y1->at(i, j);
        }
      });
    }
    {
      USE(READ, imgOut);
      USE(READWRITE, y2);
      forall<simd_exec>(0, h, [=](int j) {
        T yp1{0.0}, yp2{0.0}, tp1{0.0}, tp2{0.0};
        for (int i = w - 1; i >= 0; --i) {
          y2->at(i, j) = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
          tp2 = tp1;
          tp1 = imgOut->at(i, j);
          yp2 = yp1;
          yp1 = y2->at(i, j);
        }
      });
    }
    {
      USE(READ, y1, y2);
      USE(READWRITE, imgOut);
      forallN<exec_pol>(
        RangeSegment{0, w},
        RangeSegment{0, h},
        [=](int i, int j) {
          imgOut->at(i, j) = c2 * (y1->at(i, j) + y2->at(i, j));
        });
    }
  }
};
} // Base
} // RAJA
#endif
