// Copyright (C) 2011-2013 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PointSource.h"
#include "FiniteElement.h"
#include "GenericDofMap.h"
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <limits>
#include <memory>
#include <vector>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
PointSource::PointSource(
    std::shared_ptr<const function::FunctionSpace> V,
    const std::vector<std::pair<geometry::Point, double>> sources)
    : _function_space0(V)
{
  // Checking meshes exist
  dolfin_assert(_function_space0->mesh());

  // Copy sources
  std::vector<std::pair<geometry::Point, double>> sources_copy = sources;

  // Distribute sources
  const mesh::Mesh& mesh0 = *_function_space0->mesh();
  distribute_sources(mesh0, sources_copy);

  // Check that function space is supported
  check_space_supported(*V);
}
//-----------------------------------------------------------------------------
PointSource::PointSource(
    std::shared_ptr<const function::FunctionSpace> V0,
    std::shared_ptr<const function::FunctionSpace> V1,
    const std::vector<std::pair<geometry::Point, double>> sources)
    : _function_space0(V0), _function_space1(V1)
{
  // Check that function spaces are supported
  dolfin_assert(V0);
  dolfin_assert(V1);
  check_space_supported(*V0);
  check_space_supported(*V1);

  // Copy sources
  std::vector<std::pair<geometry::Point, double>> sources_copy = sources;

  dolfin_assert(_function_space0->mesh());
  const mesh::Mesh& mesh0 = *_function_space0->mesh();
  distribute_sources(mesh0, sources_copy);
}
//-----------------------------------------------------------------------------
PointSource::~PointSource()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PointSource::distribute_sources(
    const mesh::Mesh& mesh,
    const std::vector<std::pair<geometry::Point, double>>& sources)
{
  // Take a list of points, and assign to correct process
  const MPI_Comm mpi_comm = mesh.mpi_comm();
  const std::shared_ptr<geometry::BoundingBoxTree> tree = mesh.bounding_box_tree();

  // Collect up any points/values which are not local
  std::vector<double> remote_points;
  for (auto& s : sources)
  {
    const geometry::Point& p = s.first;
    double magnitude = s.second;

    unsigned int cell_index = tree->compute_first_entity_collision(p, mesh);
    if (cell_index == std::numeric_limits<unsigned int>::max())
    {
      remote_points.insert(remote_points.end(), p.coordinates(),
                           p.coordinates() + 3);
      remote_points.push_back(magnitude);
    }
    else
      _sources.push_back({p, magnitude});
  }

  // Send all non-local points out to all other processes
  const int mpi_size = MPI::size(mpi_comm);
  const int mpi_rank = MPI::rank(mpi_comm);
  std::vector<std::vector<double>> remote_points_all(mpi_size);
  MPI::all_gather(mpi_comm, remote_points, remote_points_all);

  // Flatten result back into remote_points vector
  // All processes will have the same data
  remote_points.clear();
  for (auto& q : remote_points_all)
    remote_points.insert(remote_points.end(), q.begin(), q.end());

  // Go through all received points, looking for any which are local
  std::vector<int> point_count;
  for (auto q = remote_points.begin(); q != remote_points.end(); q += 4)
  {
    geometry::Point p(*q, *(q + 1), *(q + 2));
    unsigned int cell_index = tree->compute_first_entity_collision(p, mesh);
    point_count.push_back(cell_index
                          != std::numeric_limits<unsigned int>::max());
  }

  // Send out the results of the search to all processes
  std::vector<std::vector<int>> point_count_all(mpi_size);
  MPI::all_gather(mpi_comm, point_count, point_count_all);

  // Check the point exists on some process, and prioritise lower rank
  for (std::size_t i = 0; i < point_count.size(); ++i)
  {
    bool found = false;
    for (int j = 0; j < mpi_size; ++j)
    {
      // Clear higher ranked 'finds' if already found on a lower rank
      // process
      if (found)
        point_count_all[j][i] = 0;

      if (point_count_all[j][i] != 0)
        found = true;
    }
    if (!found)
    {
      log::dolfin_error(
          "PointSource.cpp", "apply point source to vector",
          "The point is outside of the domain"); // (%s)", p.str().c_str());
    }
  }

  unsigned int i = 0;
  for (auto q = remote_points.begin(); q != remote_points.end(); q += 4)
  {
    if (point_count_all[mpi_rank][i] == 1)
    {
      const geometry::Point p(*q, *(q + 1), *(q + 2));
      double val = *(q + 3);
      _sources.push_back({p, val});
    }
    ++i;
  }
}
//-----------------------------------------------------------------------------
void PointSource::apply(la::PETScVector& b)
{
  // Applies local point sources.
  dolfin_assert(_function_space0);
  if (_function_space1)
  {
    log::dolfin_error("PointSource.cpp", "apply point source to vector",
                 "Can only have one function space for a vector");
  }
  log::log(PROGRESS, "Applying point source to right-hand side vector.");

  dolfin_assert(_function_space0->mesh());
  const mesh::Mesh& mesh = *_function_space0->mesh();
  const std::shared_ptr<geometry::BoundingBoxTree> tree = mesh.bounding_box_tree();

  // Variables for cell information
  std::vector<double> coordinate_dofs;

  // Variables for evaluating basis
  dolfin_assert(_function_space0->element());
  const std::size_t rank = _function_space0->element()->value_rank();
  std::size_t size_basis = 1;
  for (std::size_t i = 0; i < rank; ++i)
    size_basis *= _function_space0->element()->value_dimension(i);
  std::size_t dofs_per_cell = _function_space0->element()->space_dimension();
  std::vector<double> basis(size_basis);
  std::vector<double> values(dofs_per_cell);

  // Variables for adding local information to vector
  double basis_sum;

  for (auto& s : _sources)
  {
    geometry::Point& p = s.first;
    double magnitude = s.second;

    unsigned int cell_index = tree->compute_first_entity_collision(p, mesh);

    // Create cell
    mesh::Cell cell(mesh, static_cast<std::size_t>(cell_index));
    cell.get_coordinate_dofs(coordinate_dofs);

    // Evaluate all basis functions at the point()

    for (std::size_t i = 0; i < dofs_per_cell; ++i)
    {
      _function_space0->element()->evaluate_basis(
          i, basis.data(), p.coordinates(), coordinate_dofs.data(), -1);

      basis_sum = 0.0;
      for (const auto& v : basis)
        basis_sum += v;
      values[i] = magnitude * basis_sum;
    }

    // Compute local-to-global mapping
    dolfin_assert(_function_space0->dofmap());
    auto dofs = _function_space0->dofmap()->cell_dofs(cell.index());

    // Add values to vector
    b.add_local(values.data(), dofs_per_cell, dofs.data());
  }

  b.apply();
}
//-----------------------------------------------------------------------------
void PointSource::apply(la::PETScMatrix& A)
{
  // Applies local point sources.
  dolfin_assert(_function_space0);

  if (!_function_space1)
    _function_space1 = _function_space0;
  dolfin_assert(_function_space1);

  // Currently only works if V0 and V1 are the same
  if (_function_space0->element()->signature()
      != _function_space1->element()->signature())
  {
    log::dolfin_error("PointSource.cpp", "apply point source to matrix",
                 "The elemnts are different. Not currently implemented");
  }

  std::shared_ptr<const function::FunctionSpace> V0 = _function_space0;
  std::shared_ptr<const function::FunctionSpace> V1 = _function_space1;

  log::log(PROGRESS, "Applying point source to matrix.");

  dolfin_assert(V0->mesh());
  dolfin_assert(V0->element());
  dolfin_assert(V1->element());
  dolfin_assert(V0->dofmap());
  dolfin_assert(V1->dofmap());

  const auto mesh = V0->mesh();

  const std::shared_ptr<geometry::BoundingBoxTree> tree = mesh->bounding_box_tree();
  unsigned int cell_index;

  // Variables for cell information
  std::vector<double> coordinate_dofs;

  // Variables for evaluating basis
  const std::size_t rank = V0->element()->value_rank();
  std::size_t size_basis;
  double basis_sum0;
  double basis_sum1;

  std::size_t num_sub_spaces = V0->element()->num_sub_elements();
  // Making sure scalar function space has 1 sub space
  if (num_sub_spaces == 0)
    num_sub_spaces = 1;
  std::size_t dofs_per_cell0
      = V0->element()->space_dimension() / num_sub_spaces;
  std::size_t dofs_per_cell1
      = V1->element()->space_dimension() / num_sub_spaces;

  // Calculates size of basis
  size_basis = 1;
  for (std::size_t i = 0; i < rank; ++i)
    size_basis *= V0->element()->value_dimension(i);

  std::vector<double> basis0(size_basis);
  std::vector<double> basis1(size_basis);

  // Values vector for all sub spaces
  boost::multi_array<double, 2> values(
      boost::extents[dofs_per_cell0 * num_sub_spaces]
                    [dofs_per_cell1 * num_sub_spaces]);
  // Values vector for one subspace.
  boost::multi_array<double, 2> values_sub(
      boost::extents[dofs_per_cell0][dofs_per_cell1]);

  // Runs some checks on vector or mixed function spaces
  if (num_sub_spaces > 1)
  {
    for (std::size_t n = 0; n < num_sub_spaces; ++n)
    {
      // Doesn't work for mixed function spaces with different
      // elements.
      if (V0->sub({0})->element()->signature()
          != V0->sub({n})->element()->signature())
      {
        log::dolfin_error(
            "PointSource.cpp", "apply point source to matrix",
            "The mixed elements are not the same. Not currently implemented");
      }

      if (V0->sub({n})->element()->num_sub_elements() > 1)
      {
        log::dolfin_error("PointSource.cpp", "apply point source to matrix",
                     "Have vector elements. Not currently implemented");
      }
    }
  }

  for (auto& s : _sources)
  {
    geometry::Point& p = s.first;
    double magnitude = s.second;

    // Create cell
    cell_index = tree->compute_first_entity_collision(p, *mesh);
    mesh::Cell cell(*mesh, static_cast<std::size_t>(cell_index));

    // Cell information
    cell.get_coordinate_dofs(coordinate_dofs);

    // Calculate values with magnitude*basis_sum_0*basis_sum_1
    for (std::size_t i = 0; i < dofs_per_cell0; ++i)
    {
      V0->element()->evaluate_basis(i, basis0.data(), p.coordinates(),
                                    coordinate_dofs.data(), -1);

      for (std::size_t j = 0; j < dofs_per_cell0; ++j)
      {
        V1->element()->evaluate_basis(j, basis1.data(), p.coordinates(),
                                      coordinate_dofs.data(), -1);

        basis_sum0 = 0.0;
        basis_sum1 = 0.0;
        for (const auto& v : basis0)
          basis_sum0 += v;
        for (const auto& v : basis1)
          basis_sum1 += v;

        values_sub[i][j] = magnitude * basis_sum0 * basis_sum1;
      }
    }

    // If scalar function space, values = values_sub
    if (num_sub_spaces < 2)
      values = values_sub;
    // If vector function space with repeated sub spaces, calculates
    // the values_sub for a sub space and then manipulates values
    // matrix for all sub_spaces.
    else
    {
      int ii, jj;
      for (std::size_t k = 0; k < num_sub_spaces; ++k)
      {
        ii = 0;
        for (std::size_t i = k * dofs_per_cell0; i < dofs_per_cell0 * (k + 1);
             ++i)
        {
          jj = 0;
          for (std::size_t j = k * dofs_per_cell1; j < dofs_per_cell1 * (k + 1);
               ++j)
          {
            values[i][j] = values_sub[ii][jj];
            jj += 1;
          }
          ii += 1;
        }
      }
    }

    // Compute local-to-global mapping
    auto dofs0 = V0->dofmap()->cell_dofs(cell.index());
    auto dofs1 = V1->dofmap()->cell_dofs(cell.index());

    // Add values to matrix
    A.add_local(values.data(), dofs_per_cell0 * num_sub_spaces, dofs0.data(),
                dofs_per_cell1 * num_sub_spaces, dofs1.data());
  }

  A.apply(la::PETScMatrix::AssemblyType::FINAL);
}
//-----------------------------------------------------------------------------
void PointSource::check_space_supported(const function::FunctionSpace& V)
{
  dolfin_assert(V.element());
  if (V.element()->value_rank() > 1)
  {
    log::dolfin_error("PointSource.cpp", "create point source",
                 "function::Function must have rank 0 or 1");
  }
}
//-----------------------------------------------------------------------------
