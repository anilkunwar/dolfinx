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
import dolfin
from numba import cfunc, types, carray


@pytest.fixture
def mesh():
    return dolfin.UnitCubeMesh(dolfin.MPI.comm_world, 8, 8, 8)


@pytest.fixture
def V(mesh):
    return dolfin.FunctionSpace(mesh, 'CG', 1)


# Define a decorator for dolfin numba expressions
def numba_expression(func):
    c_sig = types.void(types.CPointer(types.double),
                       types.CPointer(types.double),
                       types.intc, types.intc, types.intc)
    return cfunc(c_sig, nopython=True)(func)


def test_expression_attach():

    @numba_expression
    def my_callback(value, x, np, gdim, vdim):
        x_array = carray(x, (np, gdim))
        val_array = carray(value, (np, vdim))
        val_array[:, 0] = x_array[:, 0] + x_array[:, 1]

    # print("Test: ", my_callback.address)
    from dolfin import cpp
    e = cpp.function.Expression([0], my_callback.address)

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

    # print(my_callback.inspect_llvm())


def test_vector_expression():

    @numba_expression
    def vec_fun(_v, _x, np, gdim, vdim):
        x = carray(_x, (np, gdim))
        vals = carray(_v, (np, vdim))
        vals[:, 0] = x[:, 0] + x[:, 1]
        vals[:, 1] = x[:, 0] - x[:, 1]

    from dolfin import cpp
    e = cpp.function.Expression([1], vec_fun.address)

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
