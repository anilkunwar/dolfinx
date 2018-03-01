// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "SpecialFunctions.h"
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>

using namespace dolfin::function;

//-----------------------------------------------------------------------------
MeshCoordinates::MeshCoordinates(std::shared_ptr<const mesh::Mesh> mesh)
    : Expression({mesh->geometry().dim()}), _mesh(mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MeshCoordinates::eval(Eigen::Ref<Eigen::VectorXd> values,
                           Eigen::Ref<const Eigen::VectorXd> x,
                           const ufc::cell& cell) const
{
  dolfin_assert(_mesh);
  dolfin_assert(cell.geometric_dimension == _mesh->geometry().dim());
  dolfin_assert((unsigned int)x.size() == _mesh->geometry().dim());

  for (std::size_t i = 0; i < cell.geometric_dimension; ++i)
    values[i] = x[i];
}
//-----------------------------------------------------------------------------
FacetArea::FacetArea(std::shared_ptr<const mesh::Mesh> mesh)
    : Expression({}), _mesh(mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void FacetArea::eval(Eigen::Ref<Eigen::VectorXd> values,
                     Eigen::Ref<const Eigen::VectorXd> x,
                     const ufc::cell& cell) const
{
  dolfin_assert(_mesh);
  dolfin_assert(cell.geometric_dimension == _mesh->geometry().dim());

  if (cell.local_facet >= 0)
  {
    mesh::Cell c(*_mesh, cell.index);
    values[0] = c.facet_area(cell.local_facet);
  }
  else
  {
    // not_on_boundary
    values[0] = 0.0;
  }
}
//-----------------------------------------------------------------------------
