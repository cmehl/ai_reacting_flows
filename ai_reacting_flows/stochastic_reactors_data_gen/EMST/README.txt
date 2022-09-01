# Compilation of EMST library:
gfortran -shared -fPIC -c emst_subs.f
f2py -c -m emst_mixing emst.f emst_subs.o

Remark:
In the python code, numpy arrays have to be float32 in order to be compatible with the FORTRAN code
Also the argument order="F" is necessary to ensure FORTRAN memory contiguity
