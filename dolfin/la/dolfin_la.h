#ifndef __DOLFIN_LA_H
#define __DOLFIN_LA_H

// DOLFIN la interface

// Note that the order is important!

#include <dolfin/la/LinearAlgebraObject.h>
#include <dolfin/la/GenericLinearOperator.h>

#include <dolfin/la/GenericTensor.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/VectorSpaceBasis.h>

#include <dolfin/la/PETScOptions.h>
#include <dolfin/la/PETScObject.h>
#include <dolfin/la/PETScBaseMatrix.h>

#include <dolfin/la/PETScMatrix.h>

#include <dolfin/la/PETScKrylovSolver.h>
#include <dolfin/la/PETScLUSolver.h>

#include <dolfin/la/CoordinateMatrix.h>
#include <dolfin/la/PETScVector.h>

#include <dolfin/la/TensorLayout.h>
#include <dolfin/la/SparsityPattern.h>

#include <dolfin/la/IndexMap.h>

#include <dolfin/la/SLEPcEigenSolver.h>
#include <dolfin/la/Scalar.h>
#include <dolfin/la/solve.h>
#include <dolfin/la/LinearOperator.h>

#endif
