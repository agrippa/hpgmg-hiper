//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#ifdef _OPENMP
#include "./timers/omp.c"
#elif USE_MPI
#include "./timers/mpi.c"
#elif USE_UPCXX
#include "./timers/gasnet.c"
#else
#error You need to include a custom timer routine
#endif
