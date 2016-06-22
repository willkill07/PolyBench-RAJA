# PolyBench/C-RAJA 0.1.0

PolyBench is a benchmark suite of 30 numerical computations with
static control flow, extracted from operations in various application
domains (linear algebra computations, image processing, physics
simulation, dynamic programming, statistics, etc.).

Copyright (c) 2016 Lawrence Livermore National Laboratory

Copyright (c) 2016 University of Delaware

### Contact:

* William Killian (<killian4@llnl.gov>) or (<killian@udel.edu>)

## PolyBench/C 4.2.1

PolyBench was originally written by Louis-Noel Pouchet.

Copyright (c) 2011-2016 the Ohio State University.

### PolyBench/C Contacts:

* Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
* Tomofumi Yuki <tomofumi.yuki@inria.fr>

## Changes from PolyBench/C

* Using runtime multi-dimensional arrays with index computation lifted through templates
* Base `PolyBenchKernel` abstract class for implementing kernels
* Instrumentation can be extended or changed by modifying `src/PolyBenchKernel.cpp`
* **RAJA** is included as a submodule of this repository
* **RAJA Portability Layer** versions of each kernel are included as well as C++ versions of the original C kernels
* Dataset sizes are no longer stored with source code. They can be extracted from `common/polybench.spec`
* Execution script where user can specify problem size is included (see `runall.sh`)

## Current State

All kernels are functionally equivalent. Some are parallelized; others just use `simd_exec` or `seq_exec`

## Available benchmarks

| Benchmark | Description |
| --------- | ----------- |
| 2mm | 2 Matrix Multiplications (alpha * A * B * C + beta * D) |
| 3mm | 3 Matrix Multiplications ((A * B) * (C * D)) |
| adi | Alternating Direction Implicit solver |
| atax | Matrix Transpose and Vector Multiplication |
| bicg | BiCG Sub Kernel of BiCGStab Linear Solver |
| cholesky | Cholesky Decomposition |
| correlation | Correlation Computation |
| covariance | Covariance Computation |
| deriche | Edge detection filter |
| doitgen | Multi-resolution analysis kernel (MADNESS) |
| durbin | Toeplitz system solver |
| fdtd-2d | 2-D Finite Different Time Domain Kernel |
| gemm | Matrix-multiply C=alpha.A.B+beta.C |
| gemver | Vector Multiplication and Matrix Addition |
| gesummv | Scalar, Vector and Matrix Multiplication |
| gramschmidt | Gram-Schmidt decomposition |
| head-3d | Heat equation over 3D data domain |
| jacobi-1D | 1-D Jacobi stencil computation |
| jacobi-2D | 2-D Jacobi stencil computation |
| lu | LU decomposition |
| ludcmp | LU decomposition followed by Forward Substitution |
| mvt | Matrix Vector Product and Transpose |
| nussinov | Dynamic programming algorithm for sequence alignment |
| seidel | 2-D Seidel stencil computation |
| symm | Symmetric matrix-multiply |
| syr2k | Symmetric rank-2k update |
| syrk | Symmetric rank-k update |
| trisolv | Triangular solver |
| trmm | Triangular matrix-multiply |
