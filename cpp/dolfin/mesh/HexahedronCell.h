// Copyright (C) 2015 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CellType.h"
#include <vector>

namespace dolfin
{

/// This class implements functionality for hexahedral cell  meshes.

class HexahedronCell : public CellType
{
public:
  /// Specify cell type and facet type
  HexahedronCell() : CellType(Type::hexahedron, Type::quadrilateral) {}

  /// Check if cell is a simplex
  bool is_simplex() const { return false; }

  /// Return topological dimension of cell
  std::size_t dim() const;

  /// Return number of entities of given topological dimension
  std::size_t num_entities(std::size_t dim) const;

  /// Return number of vertices for entity of given topological dimension
  std::size_t num_vertices(std::size_t dim) const;

  /// Return orientation of the cell
  std::size_t orientation(const Cell& cell) const;

  /// Create entities e of given topological dimension from vertices v
  void create_entities(boost::multi_array<std::int32_t, 2>& e, std::size_t dim,
                       const std::int32_t* v) const;

  /// Compute (generalized) volume (area) of triangle
  double volume(const MeshEntity& triangle) const;

  /// Compute diameter of triangle
  double circumradius(const MeshEntity& triangle) const;

  /// Compute squared distance to given point (3D enabled)
  double squared_distance(const Cell& cell, const Point& point) const;

  /// Compute component i of normal of given facet with respect to the cell
  double normal(const Cell& cell, std::size_t facet, std::size_t i) const;

  /// Compute of given facet with respect to the cell
  Point normal(const Cell& cell, std::size_t facet) const;

  /// Compute normal to given cell (viewed as embedded in 3D)
  Point cell_normal(const Cell& cell) const;

  /// Compute the area/length of given facet with respect to the cell
  double facet_area(const Cell& cell, std::size_t facet) const;

  /// Order entities locally
  void
  order(Cell& cell,
        const std::vector<std::int64_t>& local_to_global_vertex_indices) const;

  /// Check whether given point collides with cell
  bool collides(const Cell& cell, const Point& point) const;

  /// Check whether given entity collides with cell
  bool collides(const Cell& cell, const MeshEntity& entity) const;

  /// Return description of cell type
  std::string description(bool plural) const;

  /// Mapping of DOLFIN/UFC vertex ordering to VTK/XDMF ordering
  std::vector<std::int8_t> vtk_mapping() const
  {
    return {0, 1, 3, 2, 4, 5, 7, 6};
  }
};
}
