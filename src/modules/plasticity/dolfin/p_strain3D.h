// Automatically generated by FFC, the FEniCS Form Compiler, version 0.3.4.
// For further information, go to http://www/fenics.org/ffc/.
// Licensed under the GNU GPL Version 2.

#ifndef __P_STRAIN3D_H
#define __P_STRAIN3D_H

#include <dolfin/Mesh.h>
#include <dolfin/Cell.h>
#include <dolfin/Point.h>
#include <dolfin/AffineMap.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/FiniteElementSpec.h>
#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>
#include <dolfin/Functional.h>
#include <dolfin/FEM.h>

namespace dolfin { namespace p_strain3D {

/// This class contains the form to be evaluated, including
/// contributions from the interior and boundary of the domain.

class BilinearForm : public dolfin::BilinearForm
{
public:

  class TestElement;

  class TrialElement;

  BilinearForm();
  

  bool interior_contribution() const;

  void eval(real block[], const AffineMap& map) const;

  bool boundary_contribution() const;

  void eval(real block[], const AffineMap& map, unsigned int facet) const;

};

class BilinearForm::TestElement : public dolfin::FiniteElement
{
public:

  TestElement() : dolfin::FiniteElement(), tensordims(0), subelements(0)
  {
    tensordims = new unsigned int [1];
    tensordims[0] = 6;

    // Element is simple, don't need to initialize subelements
  }

  ~TestElement()
  {
    if ( tensordims ) delete [] tensordims;
    if ( subelements )
    {
      for (unsigned int i = 0; i < elementdim(); i++)
        delete subelements[i];
      delete [] subelements;
    }
  }

  inline unsigned int spacedim() const
  {
    return 6;
  }

  inline unsigned int shapedim() const
  {
    return 3;
  }

  inline unsigned int tensordim(unsigned int i) const
  {
    dolfin_assert(i < 1);
    return tensordims[i];
  }

  inline unsigned int elementdim() const
  {
    return 1;
  }

  inline unsigned int rank() const
  {
    return 1;
  }

  void nodemap(int nodes[], const Cell& cell, const Mesh& mesh) const
  {
    nodes[0] = cell.index();
    int offset = mesh.topology().size(3);
    nodes[1] = offset + cell.index();
    offset = offset + mesh.topology().size(3);
    nodes[2] = offset + cell.index();
    offset = offset + mesh.topology().size(3);
    nodes[3] = offset + cell.index();
    offset = offset + mesh.topology().size(3);
    nodes[4] = offset + cell.index();
    offset = offset + mesh.topology().size(3);
    nodes[5] = offset + cell.index();
  }

  void pointmap(Point points[], unsigned int components[], const AffineMap& map) const
  {
    points[0] = map(2.500000000000000e-01, 2.500000000000000e-01, 2.500000000000000e-01);
    points[1] = map(2.500000000000000e-01, 2.500000000000000e-01, 2.500000000000000e-01);
    points[2] = map(2.500000000000000e-01, 2.500000000000000e-01, 2.500000000000000e-01);
    points[3] = map(2.500000000000000e-01, 2.500000000000000e-01, 2.500000000000000e-01);
    points[4] = map(2.500000000000000e-01, 2.500000000000000e-01, 2.500000000000000e-01);
    points[5] = map(2.500000000000000e-01, 2.500000000000000e-01, 2.500000000000000e-01);
    components[0] = 0;
    components[1] = 1;
    components[2] = 2;
    components[3] = 3;
    components[4] = 4;
    components[5] = 5;
  }

  void vertexeval(uint vertex_nodes[], unsigned int vertex, const Mesh& mesh) const
  {
    // FIXME: Temporary fix for Lagrange elements
    vertex_nodes[0] = vertex;
    int offset = mesh.topology().size(3);
    vertex_nodes[1] = offset + vertex;
    offset = offset + mesh.topology().size(3);
    vertex_nodes[2] = offset + vertex;
    offset = offset + mesh.topology().size(3);
    vertex_nodes[3] = offset + vertex;
    offset = offset + mesh.topology().size(3);
    vertex_nodes[4] = offset + vertex;
    offset = offset + mesh.topology().size(3);
    vertex_nodes[5] = offset + vertex;
  }

  const FiniteElement& operator[] (unsigned int i) const
  {
    return *this;
  }

  FiniteElement& operator[] (unsigned int i)
  {
    return *this;
  }

  FiniteElementSpec spec() const
  {
    FiniteElementSpec s("Discontinuous vector Lagrange", "tetrahedron", 0, 6);
    return s;
  }
  
private:

  unsigned int* tensordims;
  FiniteElement** subelements;

};

class BilinearForm::TrialElement : public dolfin::FiniteElement
{
public:

  TrialElement() : dolfin::FiniteElement(), tensordims(0), subelements(0)
  {
    tensordims = new unsigned int [1];
    tensordims[0] = 6;

    // Element is simple, don't need to initialize subelements
  }

  ~TrialElement()
  {
    if ( tensordims ) delete [] tensordims;
    if ( subelements )
    {
      for (unsigned int i = 0; i < elementdim(); i++)
        delete subelements[i];
      delete [] subelements;
    }
  }

  inline unsigned int spacedim() const
  {
    return 6;
  }

  inline unsigned int shapedim() const
  {
    return 3;
  }

  inline unsigned int tensordim(unsigned int i) const
  {
    dolfin_assert(i < 1);
    return tensordims[i];
  }

  inline unsigned int elementdim() const
  {
    return 1;
  }

  inline unsigned int rank() const
  {
    return 1;
  }

  void nodemap(int nodes[], const Cell& cell, const Mesh& mesh) const
  {
    nodes[0] = cell.index();
    int offset = mesh.topology().size(3);
    nodes[1] = offset + cell.index();
    offset = offset + mesh.topology().size(3);
    nodes[2] = offset + cell.index();
    offset = offset + mesh.topology().size(3);
    nodes[3] = offset + cell.index();
    offset = offset + mesh.topology().size(3);
    nodes[4] = offset + cell.index();
    offset = offset + mesh.topology().size(3);
    nodes[5] = offset + cell.index();
  }

  void pointmap(Point points[], unsigned int components[], const AffineMap& map) const
  {
    points[0] = map(2.500000000000000e-01, 2.500000000000000e-01, 2.500000000000000e-01);
    points[1] = map(2.500000000000000e-01, 2.500000000000000e-01, 2.500000000000000e-01);
    points[2] = map(2.500000000000000e-01, 2.500000000000000e-01, 2.500000000000000e-01);
    points[3] = map(2.500000000000000e-01, 2.500000000000000e-01, 2.500000000000000e-01);
    points[4] = map(2.500000000000000e-01, 2.500000000000000e-01, 2.500000000000000e-01);
    points[5] = map(2.500000000000000e-01, 2.500000000000000e-01, 2.500000000000000e-01);
    components[0] = 0;
    components[1] = 1;
    components[2] = 2;
    components[3] = 3;
    components[4] = 4;
    components[5] = 5;
  }

  void vertexeval(uint vertex_nodes[], unsigned int vertex, const Mesh& mesh) const
  {
    // FIXME: Temporary fix for Lagrange elements
    vertex_nodes[0] = vertex;
    int offset = mesh.topology().size(3);
    vertex_nodes[1] = offset + vertex;
    offset = offset + mesh.topology().size(3);
    vertex_nodes[2] = offset + vertex;
    offset = offset + mesh.topology().size(3);
    vertex_nodes[3] = offset + vertex;
    offset = offset + mesh.topology().size(3);
    vertex_nodes[4] = offset + vertex;
    offset = offset + mesh.topology().size(3);
    vertex_nodes[5] = offset + vertex;
  }

  const FiniteElement& operator[] (unsigned int i) const
  {
    return *this;
  }

  FiniteElement& operator[] (unsigned int i)
  {
    return *this;
  }

  FiniteElementSpec spec() const
  {
    FiniteElementSpec s("Discontinuous vector Lagrange", "tetrahedron", 0, 6);
    return s;
  }
  
private:

  unsigned int* tensordims;
  FiniteElement** subelements;

};

BilinearForm::BilinearForm() : dolfin::BilinearForm(0)
{
  // Create finite element for test space
  _test = new TestElement();

  // Create finite element for trial space
  _trial = new TrialElement();
}

// Contribution from the interior
bool BilinearForm::interior_contribution() const { return true; }

void BilinearForm::eval(real block[], const AffineMap& map) const
{
  // Compute geometry tensors
  const real G0_ = map.det;

  // Compute element tensor
  block[0] = 1.666666666666665e-01*G0_;
  block[1] = 0.000000000000000e+00;
  block[2] = 0.000000000000000e+00;
  block[3] = 0.000000000000000e+00;
  block[4] = 0.000000000000000e+00;
  block[5] = 0.000000000000000e+00;
  block[6] = 0.000000000000000e+00;
  block[7] = 1.666666666666665e-01*G0_;
  block[8] = 0.000000000000000e+00;
  block[9] = 0.000000000000000e+00;
  block[10] = 0.000000000000000e+00;
  block[11] = 0.000000000000000e+00;
  block[12] = 0.000000000000000e+00;
  block[13] = 0.000000000000000e+00;
  block[14] = 1.666666666666665e-01*G0_;
  block[15] = 0.000000000000000e+00;
  block[16] = 0.000000000000000e+00;
  block[17] = 0.000000000000000e+00;
  block[18] = 0.000000000000000e+00;
  block[19] = 0.000000000000000e+00;
  block[20] = 0.000000000000000e+00;
  block[21] = 1.666666666666665e-01*G0_;
  block[22] = 0.000000000000000e+00;
  block[23] = 0.000000000000000e+00;
  block[24] = 0.000000000000000e+00;
  block[25] = 0.000000000000000e+00;
  block[26] = 0.000000000000000e+00;
  block[27] = 0.000000000000000e+00;
  block[28] = 1.666666666666665e-01*G0_;
  block[29] = 0.000000000000000e+00;
  block[30] = 0.000000000000000e+00;
  block[31] = 0.000000000000000e+00;
  block[32] = 0.000000000000000e+00;
  block[33] = 0.000000000000000e+00;
  block[34] = 0.000000000000000e+00;
  block[35] = 1.666666666666665e-01*G0_;
}

// No contribution from the boundary
bool BilinearForm::boundary_contribution() const { return false; }

void BilinearForm::eval(real block[], const AffineMap& map, unsigned int facet) const {}

} }

#endif
