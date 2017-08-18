# AcousticBEM
AcousticBEM is a small library and example programs for solving the acoustic Helmholtz equation using the Boundary Element Method. The library is a re-implementation of a large part of the functionality of Stephen Kirkup's ABEM Fortran library. The original Fortran code and his book "The Boundary Element Method in Acoustics" are available on his website: <http://www.boundary-element-method.com/>.

## Directory Structure
All the project's code is in subdirectories of the repository. The `papers` directory contains PDF files of the aforementioned book by S. Kirkup as well as two later papers relating to BEM in Acoustics.

The `Fortran` directory contains Kirkup's original Fortan 77 code. There are rudimentary `makefile`s for building the executables.

The `Python` subdirectory contains all the Python library code.

The `C` subdirectory contains C implementations of the discrete integral operators. These methods are accessible via Python native method invocation. The Python files allow configuring using these optimized versions over the Python implementation and optimized is the default setting of the code as it is checked in.

The `Jupyter` subdirectory contains a number of Jupyter notebooks that implement the example programs from the original Fortan library. The checked in versions contain results and can be opened directly from the Github web page.

## Building

For AcousticBEM the only code requiring compilation are the integration methods and the methods implementing the discrete integral operators (L, N, M, Mt in 2D, 3D, and RAD) variations. The Hankel functions used in this code are provided by the GNU Scientific Library (GSL). On an Ubuntu system this can be installed via apt-get and that's should be the only dependency requiring attention. 

The original Fortran code also requires the GSL library, which has Fortran bindings called fgsl. The makefiles require the fgsl.mod and libfgsl.a files in a subdirectory under the 'Fortran' directory, called 'fgsl'.
