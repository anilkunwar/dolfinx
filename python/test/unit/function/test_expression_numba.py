"""Unit tests for Expression using Numba"""

# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Benjamin Kehlet 2012

import pytest
from dolfin import (interpolate, UnitCubeMesh, FunctionSpace, VectorFunctionSpace,
                    Cells, MPI, cpp)
from numba import cfunc, types, carray


@pytest.fixture
def mesh():
    return UnitCubeMesh(MPI.comm_world, 2, 2, 2)


@pytest.fixture
def Q(mesh):
    return FunctionSpace(mesh, 'CG', 1)


@pytest.fixture
def V(mesh):
    return VectorFunctionSpace(mesh, 'CG', 1)


# Define a decorator for dolfin numba expressions
def numba_expression(func):
    c_sig = types.void(types.CPointer(types.double),
                       types.CPointer(types.double),
                       types.intc, types.intc, types.intc)
    return cfunc(c_sig, nopython=True)(func)


def test_scalar_expression(Q):

    @numba_expression
    def my_callback(value, x, np, gdim, vdim):
        x_array = carray(x, (np, gdim))
        val_array = carray(value, (np, vdim))
        val_array[:, 0] = x_array[:, 0] + x_array[:, 1]

    # print("Test: ", my_callback.address)
    e = cpp.function.Expression([], my_callback.address)

    import numpy as np
    vals = np.zeros([2, 1])
    x = np.array([[3, 10], [1, 3]])
    print(vals.shape)
    print(vals)
    print("-----")
    print(x.shape)
    print(x)

    e.eval(vals, x)
    print("Test2: ", vals)
    assert vals[0] == 13.0
    assert vals[1] == 4.0

    F = interpolate(e, Q)
    assert np.isclose(sum(F.vector().get_local()), 27.0)


def test_vector_expression(V):

    @numba_expression
    def vec_fun(_v, _x, np, gdim, vdim):
        x = carray(_x, (np, gdim))
        vals = carray(_v, (np, vdim))
        vals[:, 0] = x[:, 0] + x[:, 1]
        vals[:, 1] = x[:, 0] - x[:, 1]
        vals[:, 2] = 1.0

    from dolfin import cpp
    e = cpp.function.Expression([3], vec_fun.address)

    import numpy as np
    vals = np.zeros([2, 2])
    x = np.array([[3, 10], [1, 3]])
    print(vals.shape)
    print(vals)
    print("-----")
    print(x.shape)
    print(x)

    e.eval(vals, x)
    print("Test2: ", vals)

    F = interpolate(e, V)

    for c in Cells(V.mesh()):
        p = c.midpoint().array()
        val = F(p)
        assert np.isclose(val[0][0], p[0] + p[1])
        assert np.isclose(val[0][1], p[0] - p[1])
        assert np.isclose(val[0][2], 1.0)
