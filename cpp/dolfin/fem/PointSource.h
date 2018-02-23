// Copyright (C) 2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/geometry/Point.h>
#include <memory>
#include <utility>
#include <vector>

namespace dolfin
{

// Forward declarations
class FunctionSpace;
class PETScMatrix;
class PETScVector;
class Mesh;

/// This class provides an easy mechanism for adding a point
/// quantities (Dirac delta function) to variational problems. The
/// associated function space must be scalar in order for the inner
/// product with the (scalar) Dirac delta function to be well
/// defined. For each of the constructors, Points passed to
/// PointSource will be copied.
///
/// Note: the interface to this class will likely change.

class PointSource
{
public:
  /// Create point sources at given points of given magnitudes
  PointSource(std::shared_ptr<const FunctionSpace> V,
              const std::vector<std::pair<Point, double>> sources);

  /// Create point sources at given points of given magnitudes
  PointSource(std::shared_ptr<const FunctionSpace> V0,
              std::shared_ptr<const FunctionSpace> V1,
              const std::vector<std::pair<Point, double>> sources);

  /// Destructor
  ~PointSource();

  /// Apply (add) point source to right-hand side vector
  void apply(PETScVector& b);

  /// Apply (add) point source to matrix
  void apply(PETScMatrix& A);

private:
  // FIXME: This should probably be static
  // Collective MPI method to distribute sources to correct processes
  void distribute_sources(const Mesh& mesh,
                          const std::vector<std::pair<Point, double>>& sources);

  // Check that function space is scalar
  static void check_space_supported(const FunctionSpace& V);

  // The function spaces
  std::shared_ptr<const FunctionSpace> _function_space0;
  std::shared_ptr<const FunctionSpace> _function_space1;

  // Source term - pair of points and magnitude
  std::vector<std::pair<Point, double>> _sources;
};
}
