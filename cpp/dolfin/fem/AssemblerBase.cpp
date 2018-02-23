// Copyright (C) 2007-2014 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <array>
#include <memory>
#include <vector>

#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/la/IndexMap.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>

#include "AssemblerBase.h"
#include "FiniteElement.h"
#include "Form.h"
#include "GenericDofMap.h"
#include "SparsityPatternBuilder.h"
#include "utils.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void AssemblerBase::init_global_tensor(PETScVector& x, const Form& a)
{
  fem::init(x, a);
  if (!add_values)
    x.zero();
}
//-----------------------------------------------------------------------------
void AssemblerBase::init_global_tensor(PETScMatrix& A, const Form& a)
{
  fem::init(A, a);
  if (!add_values)
    A.zero();
}
//-----------------------------------------------------------------------------
void AssemblerBase::check(const Form& a)
{
  dolfin_assert(a.ufc_form());

  // Check the form
  a.check();

  // Extract mesh and coefficients
  dolfin_assert(a.mesh());
  const Mesh& mesh = *(a.mesh());
  const std::vector<std::shared_ptr<const GenericFunction>> coefficients
      = a.coefficients();

  // Check ghost mode for interior facet integrals in parallel
  if (a.ufc_form()->has_interior_facet_integrals()
      && MPI::size(mesh.mpi_comm()) > 1)
  {
    std::string ghost_mode = mesh.ghost_mode();
    if (!(ghost_mode == "shared_vertex" || ghost_mode == "shared_facet"))
    {
      dolfin_error("AssemblerBase.cpp", "assemble form",
                   "Incorrect mesh ghost mode \"%s\" (expected "
                   "\"shared_vertex\" or \"shared_facet\" for "
                   "interior facet integrals in parallel)",
                   ghost_mode.c_str());
    }
  }

  // Check that we get the correct number of coefficients
  if (coefficients.size() != a.num_coefficients())
  {
    dolfin_error("AssemblerBase.cpp", "assemble form",
                 "Incorrect number of coefficients (got %d but expecting %d)",
                 coefficients.size(), a.num_coefficients());
  }

  // Check that all coefficients have valid value dimensions
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
    if (!coefficients[i])
    {
      dolfin_error("AssemblerBase.cpp", "assemble form",
                   "Coefficient number %d (\"%s\") has not been set", i,
                   a.coefficient_name(i).c_str());
    }

    // unique_ptr deletes its object when it exits its scope
    std::unique_ptr<ufc::finite_element> fe(
        a.ufc_form()->create_finite_element(i + a.rank()));

    // Checks out-commented since they only work for Functions, not
    // Expressions
    const std::size_t r = coefficients[i]->value_rank();
    const std::size_t fe_r = fe->value_rank();
    if (fe_r != r)
    {
      dolfin_error(
          "AssemblerBase.cpp", "assemble form",
          "Invalid value rank for coefficient %d (got %d but expecting %d). \
You might have forgotten to specify the value rank correctly in an Expression subclass",
          i, r, fe_r);
    }

    for (std::size_t j = 0; j < r; ++j)
    {
      const std::size_t dim = coefficients[i]->value_dimension(j);
      const std::size_t fe_dim = fe->value_dimension(j);
      if (dim != fe_dim)
      {
        dolfin_error(
            "AssemblerBase.cpp", "assemble form",
            "Invalid value dimension %d for coefficient %d (got %d but expecting %d). \
You might have forgotten to specify the value dimension correctly in an Expression subclass",
            j, i, dim, fe_dim);
      }
    }
  }

  // Check that the coordinate cell matches the mesh
  std::unique_ptr<ufc::finite_element> coordinate_element(
      a.ufc_form()->create_coordinate_finite_element());
  dolfin_assert(coordinate_element);
  dolfin_assert(coordinate_element->value_rank() == 1);
  if (coordinate_element->value_dimension(0) != mesh.geometry().dim())
  {
    dolfin_error("AssemblerBase.cpp", "assemble form",
                 "Geometric dimension of Mesh does not match value shape of "
                 "coordinate element in form");
  }

  // Check that the coordinate element degree matches the mesh degree
  if (coordinate_element->degree() != mesh.geometry().degree())
  {
    dolfin_error("AssemblerBase.cpp", "assemble form",
                 "Mesh geometry degree does not match degree of coordinate "
                 "element in form");
  }

  std::map<CellType::Type, ufc::shape> dolfin_to_ufc_shapes
      = {{CellType::Type::interval, ufc::shape::interval},
         {CellType::Type::triangle, ufc::shape::triangle},
         {CellType::Type::tetrahedron, ufc::shape::tetrahedron},
         {CellType::Type::quadrilateral, ufc::shape::quadrilateral},
         {CellType::Type::hexahedron, ufc::shape::hexahedron}};

  auto cell_type_pair = dolfin_to_ufc_shapes.find(mesh.type().cell_type());
  dolfin_assert(cell_type_pair != dolfin_to_ufc_shapes.end());
  if (coordinate_element->cell_shape() != cell_type_pair->second)
  {
    dolfin_error("AssemblerBase.cpp", "assemble form",
                 "Mesh cell type does not match cell type of UFC form");
  }
}
//-----------------------------------------------------------------------------
std::string AssemblerBase::progress_message(std::size_t rank,
                                            std::string integral_type)
{
  std::stringstream s;
  s << "Assembling ";

  switch (rank)
  {
  case 0:
    s << "scalar value over ";
    break;
  case 1:
    s << "vector over ";
    break;
  case 2:
    s << "matrix over ";
    break;
  default:
    s << "rank " << rank << " tensor over ";
    break;
  }

  s << integral_type;

  return s.str();
}
//-----------------------------------------------------------------------------
