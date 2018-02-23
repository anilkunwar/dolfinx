// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Expression.h"
#include <Eigen/Dense>
#include <dolfin/log/Event.h>
#include <memory>

namespace dolfin
{

class Mesh;

/// This Function represents the mesh coordinates on a given mesh.
class MeshCoordinates : public Expression
{
public:
  /// Constructor
  explicit MeshCoordinates(std::shared_ptr<const Mesh> mesh);

  /// Evaluate function
  void eval(Eigen::Ref<Eigen::VectorXd> values,
            Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const;

private:
  // The mesh
  std::shared_ptr<const Mesh> _mesh;
};

/// This function represents the area/length of a cell facet on a
/// given mesh.
class FacetArea : public Expression
{
public:
  /// Constructor
  explicit FacetArea(std::shared_ptr<const Mesh> mesh);

  /// Evaluate function
  void eval(Eigen::Ref<Eigen::VectorXd> values,
            Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const;

private:
  // The mesh
  std::shared_ptr<const Mesh> _mesh;

  // Warning when evaluating on cells
  mutable Event not_on_boundary;
};
}
