// Copyright (C) 2012 Fredrik Valdmanis
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Benjamin Kehlet, 2012
//
// First added:  2012-05-23
// Last changed: 2012-06-14

#ifndef __VTKPLOTTER_H
#define __VTKPLOTTER_H

#ifdef HAS_VTK

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPointSet.h>
#include <vtkPolyDataAlgorithm.h>
#include <vtkWarpScalar.h>
#include <vtkWarpVector.h>
#include <vtkGlyph3D.h>
#include <vtkGeometryFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkScalarBarActor.h>

#include <dolfin/mesh/Mesh.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/Expression.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  // FIXME: Use forward declarations to avoid inclusion of .h files in .h files

  // Forward declarations
  class PlottableExpression;

  /// This class enables visualization of various DOLFIN entities.
  /// It supports visualization of meshes, functions, expressions, boundary
  /// conditions and mesh functions.
  /// The plotter has several parameters that the user can set and adjust to
  /// affect the appearance and behavior of the plot.
  ///
  /// A plotter can be created and used in the following way:
  ///
  ///   Mesh mesh = ...;
  ///   VTKPlotter plotter(mesh);
  ///   plotter.plot();
  ///
  /// Parameters can be adjusted at any time and will take effect on the next
  /// call to the plot() method. The following parameters exist:
  ///
  /// ============= ============ =============== =================================
  ///  Name          Value type   Default value              Description
  /// ============= ============ =============== =================================
  ///  mode           String        "auto"        For vector valued functions,
  ///                                             this parameter may be set to
  ///                                             "warp" to enable vector warping
  ///                                             visualization
  ///  interactive    Boolean         True        Enable/disable interactive mode
  ///                                             for the rendering window.
  ///                                             For repeated plots of the same
  ///                                             object (animated plots), this
  ///                                             parameter must be set to false
  ///  wireframe      Boolean     True for        Enable/disable wireframe
  ///                             meshes, else    rendering of the object
  ///                             false
  ///  title          String      Inherited       The title of the rendering
  ///                             from the        window
  ///                             name/label of
  ///                             the object
  ///  scale          Double      1.0             Adjusts the scaling of the
  ///                                             warping and glyphs
  ///  scalarbar      Boolean     False for       Hide/show the colormapping bar
  ///                             meshes, else
  ///                             true
  /// ============= ============ =============== =================================
  ///
  /// The default visualization mode for the different plot types are as follows:
  ///
  /// =========================  ============================ ===================
  ///  Plot type                  Default visualization mode   Alternatives
  /// =========================  ============================ ===================
  ///  Meshes                     Wireframe rendering           None
  ///  2D scalar functions        Scalar warping                None
  ///  3D scalar functions        Color mapping                 None
  ///  2D/3D vector functions     Glyphs (vector arrows)        Vector warping
  /// =========================  ============================ ===================
  ///
  /// Expressions and boundary conditions are also visualized according to the
  /// above table.

  class VTKPlotter : public Variable
  {
  public:

    /// Copy constructor
    VTKPlotter(const VTKPlotter& plotter);

    /// Create plotter for a mesh
    explicit VTKPlotter(boost::shared_ptr<const Mesh> mesh);

    /// Create plotter for a function
    explicit VTKPlotter(boost::shared_ptr<const Function> function);

    /// Create plotter for an expression
    explicit VTKPlotter(boost::shared_ptr<const PlottableExpression> expression);

    /// Create plotter for an expression
    explicit VTKPlotter(boost::shared_ptr<const Expression> expression,
                        boost::shared_ptr<const Mesh> mesh);

    /// Create plotter for Dirichlet B.C.
    explicit VTKPlotter(boost::shared_ptr<const DirichletBC> bc);

    /// Create plotter for an integer valued mesh function
    explicit VTKPlotter(boost::shared_ptr<const MeshFunction<uint> > mesh_function);

    /// Create plotter for a double valued mesh function
    explicit VTKPlotter(boost::shared_ptr<const MeshFunction<double> > mesh_function);

    /// Create plotter for a boolean valued mesh function
    explicit VTKPlotter(boost::shared_ptr<const MeshFunction<bool> > mesh_function);

    /// Destructor
    ~VTKPlotter();

    /// Assignment operator
    const VTKPlotter& operator=(const VTKPlotter& plotter);

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("vtk_plotter");
      p.add("title", "Plot");
      p.add("interactive", true);
      p.add("wireframe", false);
      p.add("scalarbar", true);
      p.add("mode", "auto");
      p.add("scale", 1.0);
      p.add("hardcopy_prefix", "dolfin_plot_");
      return p;
    }

    // TODO: Add separate default parameters for each type?
    // TODO: Set title in default_parameters to "Mesh", "Expression" etc
    // as in plot.h?

    /// Default parameter values for mesh plotting
    static Parameters default_mesh_parameters()
    {
      Parameters p = default_parameters();
      p["wireframe"] = true;
      p["scalarbar"] = false;
      return p;
    }

    /// Plot the object
    void plot();

    /// Make the current plot interactive
    void interactive();

    /// Save plot to PNG file (file suffix appended automatically)
    void hardcopy(std::string filename);

    /// Return unique ID of the object to plot
    uint id() const { return _id; }

    // FIXME: Remove? Is this used?
    // Index to the last used plotter object
    static int last_used_idx;

    // The cache of plotter objects
    static std::vector<boost::shared_ptr<VTKPlotter> > plotter_cache;

  private:

    // Setup all pipeline objects and connect them. Called from all
    // constructors
    void init_pipeline();

    // Set the title parameter from the name and label of the Variable to plot
    void set_title(const std::string& name, const std::string& label);

    // Construct VTK grid from DOLFIN mesh
    void construct_vtk_grid();

    // Process mesh (including filter setup)
    void process_mesh();

    // Process scalar valued generic function (function or expression),
    // including filter setup
    void process_scalar_function();

    // Process vector valued generic function (function or expression),
    // including filter setup
    void process_vector_function();

    // Return the hover-over help text
    std::string get_helptext();

    // Keypress callback
    void keypressCallback(vtkObject* caller, 
                          long unsigned int eventId,
                          void* callData);

    // The mesh to visualize
    boost::shared_ptr<const Mesh> _mesh;

    // The (optional) function values to visualize
    boost::shared_ptr<const GenericFunction> _function;

    // The VTK grid constructed from the DOLFIN mesh
    vtkSmartPointer<vtkUnstructuredGrid> _grid;

    // The scalar warp filter
    vtkSmartPointer<vtkWarpScalar> _warpscalar;

    // The vector warp filter
    vtkSmartPointer<vtkWarpVector> _warpvector;

    // The glyph filter
    vtkSmartPointer<vtkGlyph3D> _glyphs;

    // The geometry filter
    vtkSmartPointer<vtkGeometryFilter> _geometryFilter;

    // The poly data mapper
    vtkSmartPointer<vtkPolyDataMapper> _mapper;

    // The lookup table
    vtkSmartPointer<vtkLookupTable> _lut;

    // The main actor
    vtkSmartPointer<vtkActor> _actor;

    // The renderer
    vtkSmartPointer<vtkRenderer> _renderer;

    // The render window
    vtkSmartPointer<vtkRenderWindow> _renderWindow;

    // The render window interactor for interactive sessions
    vtkSmartPointer<vtkRenderWindowInteractor> _interactor;

    // The scalar bar that gives the viewer the mapping from color to
    // scalar value
    vtkSmartPointer<vtkScalarBarActor> _scalarBar;

    // The unique ID (inherited from Variable) for the object to plot
    uint _id;

    // Counter for the automatically named hardcopies
    static int hardcopy_counter; 

  };

}

#endif

#endif
