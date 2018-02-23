// Copyright (C) 2013 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "intersect.h"
#include "MeshPointIntersection.h"
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Mesh.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
std::shared_ptr<const MeshPointIntersection>
dolfin::intersect(const Mesh& mesh, const Point& point)
{
  // Intersection is only implemented for simplex meshes
  if (!mesh.type().is_simplex())
  {
    dolfin_error("intersect.cpp", "intersect mesh and point",
                 "Intersection is only implemented for simplex meshes");
  }

  return std::shared_ptr<const MeshPointIntersection>(
      new MeshPointIntersection(mesh, point));
}
//-----------------------------------------------------------------------------
