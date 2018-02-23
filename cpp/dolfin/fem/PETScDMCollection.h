// Copyright (C) 2016 Patrick E. Farrell and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_PETSC

#include <dolfin/common/MPI.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/log/log.h>
#include <memory>
#include <petscdm.h>
#include <petscvec.h>
#include <vector>

namespace dolfin
{

class Mesh;
class FunctionSpace;
class BoundingBoxTree;

/// This class builds and stores of collection of PETSc DM objects
/// from a hierarchy of FunctionSpaces objects. The DM objects are
/// used to construct multigrid solvers via PETSc.
///
/// Warning: This classs is highly experimental and will change

class PETScDMCollection : public PETScObject
{
public:
  /// Construct PETScDMCollection from a vector of
  /// FunctionSpaces. The vector of FunctionSpaces is stored from
  /// coarse to fine.
  PETScDMCollection(
      std::vector<std::shared_ptr<const FunctionSpace>> function_spaces);

  /// Destructor
  ~PETScDMCollection();

  /// Return the ith DM objects. The coarest DM has index 0. Use
  /// i=-1 to get the DM for the finest level, i=-2 for the DM for
  /// the second finest level, etc.
  DM get_dm(int i);

  /// These are test/debugging functions that will be removed
  void check_ref_count() const;

  /// Debugging use - to be removed
  void reset(int i);

  /// Create the interpolation matrix from the coarse to the fine
  /// space (prolongation matrix)
  static std::shared_ptr<PETScMatrix>
  create_transfer_matrix(const FunctionSpace& coarse_space,
                         const FunctionSpace& fine_space);

private:
  // Find the nearest cells to points which lie outside the domain
  static void find_exterior_points(MPI_Comm mpi_comm, const Mesh& meshc,
                                   std::shared_ptr<const BoundingBoxTree> treec,
                                   int dim, int data_size,
                                   const std::vector<double>& send_points,
                                   const std::vector<int>& send_indices,
                                   std::vector<int>& indices,
                                   std::vector<std::size_t>& cell_ids,
                                   std::vector<double>& points);

  // Pointers to functions that are used in PETSc DM call-backs
  static PetscErrorCode create_global_vector(DM dm, Vec* vec);
  static PetscErrorCode create_interpolation(DM dmc, DM dmf, Mat* mat,
                                             Vec* vec);
  static PetscErrorCode coarsen(DM dmf, MPI_Comm comm, DM* dmc);
  static PetscErrorCode refine(DM dmc, MPI_Comm comm, DM* dmf);

  // The FunctionSpaces associated with each level, starting with
  // the coarest space
  std::vector<std::shared_ptr<const FunctionSpace>> _spaces;

  // The PETSc DM objects
  std::vector<DM> _dms;
};
} // namespace dolfin

#endif
