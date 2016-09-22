//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <functional>
//------------------------------------------------------------------------------------------------------------------------------
#include "timers.h"
#include "defines.h"
#include "level.h"
#include "operators.h"
#include "mg.h"
#include "hclib_atomic.h"
//------------------------------------------------------------------------------------------------------------------------------
#define STENCIL_VARIABLE_COEFFICIENT
//------------------------------------------------------------------------------------------------------------------------------
#define MyPragma(a) _Pragma(#a)
//------------------------------------------------------------------------------------------------------------------------------

// #define PRAGMA_THREAD_ACROSS_BLOCKS(    level,b,nb     )    MyPragma(omp parallel for private(b) if(nb>1) schedule(static,1)                     )
template <typename T>
inline void parallel_across_blocks(level_type *level, int b, int nb,
        T lambda) {
    hclib::finish([&nb, &lambda] {
        hclib::loop_domain_1d loop(nb);
        hclib::forasync1D<T>(&loop, lambda, nb <= 1);
    });
}

// #define PRAGMA_THREAD_ACROSS_BLOCKS_SUM(level,b,nb,bsum)    MyPragma(omp parallel for private(b) if(nb>1) schedule(static,1) reduction(  +:bsum) )
// #define PRAGMA_THREAD_ACROSS_BLOCKS_MAX(level,b,nb,bmax)    MyPragma(omp parallel for private(b) if(nb>1) schedule(static,1) reduction(max:bmax) )


//------------------------------------------------------------------------------------------------------------------------------
// fix... make #define...
void apply_BCs(level_type * level, int x_id, int justFaces){
  #ifndef STENCIL_FUSE_BC
  // This is a failure mode if (trying to do communication-avoiding) && (BC!=BC_PERIODIC)
  apply_BCs_linear(level,x_id,justFaces);
  #endif
}
//------------------------------------------------------------------------------------------------------------------------------
// calculate Dinv?
#ifdef STENCIL_VARIABLE_COEFFICIENT
  #ifdef USE_HELMHOLTZ // variable coefficient Helmholtz ...
  #define calculate_Dinv()                                      \
  (                                                             \
    1.0 / (a*alpha[ijk] - b*h2inv*(                             \
             + beta_i[ijk        ]*( valid[ijk-1      ] - 2.0 ) \
             + beta_j[ijk        ]*( valid[ijk-jStride] - 2.0 ) \
             + beta_k[ijk        ]*( valid[ijk-kStride] - 2.0 ) \
             + beta_i[ijk+1      ]*( valid[ijk+1      ] - 2.0 ) \
             + beta_j[ijk+jStride]*( valid[ijk+jStride] - 2.0 ) \
             + beta_k[ijk+kStride]*( valid[ijk+kStride] - 2.0 ) \
          ))                                                    \
  )
  #else // variable coefficient Poisson ...
  #define calculate_Dinv()                                      \
  (                                                             \
    1.0 / ( -b*h2inv*(                                          \
             + beta_i[ijk        ]*( valid[ijk-1      ] - 2.0 ) \
             + beta_j[ijk        ]*( valid[ijk-jStride] - 2.0 ) \
             + beta_k[ijk        ]*( valid[ijk-kStride] - 2.0 ) \
             + beta_i[ijk+1      ]*( valid[ijk+1      ] - 2.0 ) \
             + beta_j[ijk+jStride]*( valid[ijk+jStride] - 2.0 ) \
             + beta_k[ijk+kStride]*( valid[ijk+kStride] - 2.0 ) \
          ))                                                    \
  )
  #endif
#else // constant coefficient case... 
  #define calculate_Dinv()          \
  (                                 \
    1.0 / (a - b*h2inv*(            \
             + valid[ijk-1      ]   \
             + valid[ijk-jStride]   \
             + valid[ijk-kStride]   \
             + valid[ijk+1      ]   \
             + valid[ijk+jStride]   \
             + valid[ijk+kStride]   \
             - 12.0                 \
          ))                        \
  )
#endif

#if defined(STENCIL_FUSE_DINV) && defined(STENCIL_FUSE_BC)
#define Dinv_ijk() calculate_Dinv() // recalculate it
#else
#define Dinv_ijk() Dinv[ijk]        // simply retriev it rather than recalculating it
#endif
//------------------------------------------------------------------------------------------------------------------------------
#ifdef STENCIL_FUSE_BC

  #ifdef STENCIL_VARIABLE_COEFFICIENT
    #ifdef USE_HELMHOLTZ // variable coefficient Helmholtz ...
    #define apply_op_ijk(x)                                                                   \
    (                                                                                         \
      a*alpha[ijk]*x[ijk]                                                                     \
      -b*h2inv*(                                                                              \
        + beta_i[ijk        ]*( valid[ijk-1      ]*( x[ijk] + x[ijk-1      ] ) - 2.0*x[ijk] ) \
        + beta_j[ijk        ]*( valid[ijk-jStride]*( x[ijk] + x[ijk-jStride] ) - 2.0*x[ijk] ) \
        + beta_k[ijk        ]*( valid[ijk-kStride]*( x[ijk] + x[ijk-kStride] ) - 2.0*x[ijk] ) \
        + beta_i[ijk+1      ]*( valid[ijk+1      ]*( x[ijk] + x[ijk+1      ] ) - 2.0*x[ijk] ) \
        + beta_j[ijk+jStride]*( valid[ijk+jStride]*( x[ijk] + x[ijk+jStride] ) - 2.0*x[ijk] ) \
        + beta_k[ijk+kStride]*( valid[ijk+kStride]*( x[ijk] + x[ijk+kStride] ) - 2.0*x[ijk] ) \
      )                                                                                       \
    )
    #else // variable coefficient Poisson ...
    #define apply_op_ijk(x)                                                                   \
    (                                                                                         \
      -b*h2inv*(                                                                              \
        + beta_i[ijk        ]*( valid[ijk-1      ]*( x[ijk] + x[ijk-1      ] ) - 2.0*x[ijk] ) \
        + beta_j[ijk        ]*( valid[ijk-jStride]*( x[ijk] + x[ijk-jStride] ) - 2.0*x[ijk] ) \
        + beta_k[ijk        ]*( valid[ijk-kStride]*( x[ijk] + x[ijk-kStride] ) - 2.0*x[ijk] ) \
        + beta_i[ijk+1      ]*( valid[ijk+1      ]*( x[ijk] + x[ijk+1      ] ) - 2.0*x[ijk] ) \
        + beta_j[ijk+jStride]*( valid[ijk+jStride]*( x[ijk] + x[ijk+jStride] ) - 2.0*x[ijk] ) \
        + beta_k[ijk+kStride]*( valid[ijk+kStride]*( x[ijk] + x[ijk+kStride] ) - 2.0*x[ijk] ) \
      )                                                                                       \
    )
    #endif
  #else  // constant coefficient case...  
    #define apply_op_ijk(x)                                \
    (                                                    \
      a*x[ijk] - b*h2inv*(                               \
        + valid[ijk-1      ]*( x[ijk] + x[ijk-1      ] ) \
        + valid[ijk-jStride]*( x[ijk] + x[ijk-jStride] ) \
        + valid[ijk-kStride]*( x[ijk] + x[ijk-kStride] ) \
        + valid[ijk+1      ]*( x[ijk] + x[ijk+1      ] ) \
        + valid[ijk+jStride]*( x[ijk] + x[ijk+jStride] ) \
        + valid[ijk+kStride]*( x[ijk] + x[ijk+kStride] ) \
                       -12.0*( x[ijk]                  ) \
      )                                                  \
    )
  #endif // variable/constant coefficient

#endif


//------------------------------------------------------------------------------------------------------------------------------
#ifndef STENCIL_FUSE_BC

  #ifdef STENCIL_VARIABLE_COEFFICIENT
    #ifdef USE_HELMHOLTZ // variable coefficient Helmholtz...
    #define apply_op_ijk(x)                               \
    (                                                     \
      a*alpha[ijk]*x[ijk]                                 \
     -b*h2inv*(                                           \
        + beta_i[ijk+1      ]*( x[ijk+1      ] - x[ijk] ) \
        + beta_i[ijk        ]*( x[ijk-1      ] - x[ijk] ) \
        + beta_j[ijk+jStride]*( x[ijk+jStride] - x[ijk] ) \
        + beta_j[ijk        ]*( x[ijk-jStride] - x[ijk] ) \
        + beta_k[ijk+kStride]*( x[ijk+kStride] - x[ijk] ) \
        + beta_k[ijk        ]*( x[ijk-kStride] - x[ijk] ) \
      )                                                   \
    )
    #else // variable coefficient Poisson...
    #define apply_op_ijk(x)                               \
    (                                                     \
      -b*h2inv*(                                          \
        + beta_i[ijk+1      ]*( x[ijk+1      ] - x[ijk] ) \
        + beta_i[ijk        ]*( x[ijk-1      ] - x[ijk] ) \
        + beta_j[ijk+jStride]*( x[ijk+jStride] - x[ijk] ) \
        + beta_j[ijk        ]*( x[ijk-jStride] - x[ijk] ) \
        + beta_k[ijk+kStride]*( x[ijk+kStride] - x[ijk] ) \
        + beta_k[ijk        ]*( x[ijk-kStride] - x[ijk] ) \
      )                                                   \
    )
    #endif
  #else  // constant coefficient case...  
    #define apply_op_ijk(x)            \
    (                                \
      a*x[ijk] - b*h2inv*(           \
        + x[ijk+1      ]             \
        + x[ijk-1      ]             \
        + x[ijk+jStride]             \
        + x[ijk-jStride]             \
        + x[ijk+kStride]             \
        + x[ijk-kStride]             \
        - x[ijk        ]*6.0         \
      )                              \
    )
  #endif // variable/constant coefficient

#endif // BCs


//------------------------------------------------------------------------------------------------------------------------------
int stencil_get_radius()    {return(1);} // replaces #define STENCIL_RADIUS         1
int stencil_is_star_shaped(){return(1);} // replaces #define STENCIL_IS_STAR_SHAPED 1
//------------------------------------------------------------------------------------------------------------------------------
void rebuild_operator(level_type * level, level_type *fromLevel, double a, double b){
  if(level->my_rank==0){fprintf(stdout,"  rebuilding operator for level...  h=%e \n ",level->h);}

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // form restriction of alpha[], beta_*[] coefficients from fromLevel
  if(fromLevel != NULL){
    restriction(level,VECTOR_ALPHA ,fromLevel,VECTOR_ALPHA ,RESTRICT_CELL  );
    restriction(level,VECTOR_BETA_I,fromLevel,VECTOR_BETA_I,RESTRICT_FACE_I);
    restriction(level,VECTOR_BETA_J,fromLevel,VECTOR_BETA_J,RESTRICT_FACE_J);
    restriction(level,VECTOR_BETA_K,fromLevel,VECTOR_BETA_K,RESTRICT_FACE_K);
  } // else case assumes alpha/beta have been set

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // exchange alpha/beta/...  (must be done before calculating Dinv)
  exchange_boundary(level,VECTOR_ALPHA ,0); // must be 0(faces,edges,corners) for CA version or 27pt
  exchange_boundary(level,VECTOR_BETA_I,0);
  exchange_boundary(level,VECTOR_BETA_J,0);
  exchange_boundary(level,VECTOR_BETA_K,0);

  level->subteam->barrier();

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // calculate Dinv, L1inv, and estimate the dominant Eigenvalue
  uint64_t _timeStart = CycleTime();
  int block;

  // double dominant_eigenvalue = -1e9;
  hclib::atomic_max_t<double> dominant_eigenvalue_atomic(-1E9);

  // PRAGMA_THREAD_ACROSS_BLOCKS_MAX(level,block,level->num_my_blocks,dominant_eigenvalue)
  parallel_across_blocks(level, block, level->num_my_blocks,
          [&level, &dominant_eigenvalue_atomic, &b, &a] (int block) {
    const int box = level->my_blocks[block].read.box;
    const int ilo = level->my_blocks[block].read.i;
    const int jlo = level->my_blocks[block].read.j;
    const int klo = level->my_blocks[block].read.k;
    const int ihi = level->my_blocks[block].dim.i + ilo;
    const int jhi = level->my_blocks[block].dim.j + jlo;
    const int khi = level->my_blocks[block].dim.k + klo;
    int i,j,k;

    box_type* lbox = (box_type *)&(level->my_boxes[box]);
    const int jStride = lbox->jStride;
    const int kStride = lbox->kStride;
    const int  ghosts = lbox->ghosts;
    const int     dim = lbox->dim;
    double h2inv = 1.0/(level->h*level->h);

    double * __restrict__ alpha  = (double *)(lbox->vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride));
    double * __restrict__ beta_i = (double *)(lbox->vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride));
    double * __restrict__ beta_j = (double *)(lbox->vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride));
    double * __restrict__ beta_k = (double *)(lbox->vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride));
    double * __restrict__   Dinv = (double *)(lbox->vectors[VECTOR_DINV  ] + ghosts*(1+jStride+kStride));
    double * __restrict__  L1inv = (double *)(lbox->vectors[VECTOR_L1INV ] + ghosts*(1+jStride+kStride));
    double * __restrict__  valid = (double *)(lbox->vectors[VECTOR_VALID ] + ghosts*(1+jStride+kStride));
    double block_eigenvalue = -1e9;

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){ 
      int ijk = i + j*jStride + k*kStride;
      #if 0
      // FIX This looks wrong, but is faster... theory is because its doing something akin to SOR
      // assumes periodic boundary conditions...
      // radius of Gershgorin disc is the sum of the absolute values of the off-diagonal elements...
      double sumAbsAij = fabs(b*h2inv*beta_i[ijk]) + fabs(b*h2inv*beta_i[ijk+      1]) +
                         fabs(b*h2inv*beta_j[ijk]) + fabs(b*h2inv*beta_j[ijk+jStride]) +
                         fabs(b*h2inv*beta_k[ijk]) + fabs(b*h2inv*beta_k[ijk+kStride]);
      // centr of Gershgorin disc is the diagonal element...
      double    Aii = a*alpha[ijk] - b*h2inv*( 
                                       -beta_i[ijk]-beta_i[ijk+      1] 
                                       -beta_j[ijk]-beta_j[ijk+jStride] 
                                       -beta_k[ijk]-beta_k[ijk+kStride] 
                                     );
      #endif
      #if 1
      // radius of Gershgorin disc is the sum of the absolute values of the off-diagonal elements...
      double sumAbsAij = fabs(b*h2inv) * (
                      fabs( beta_i[ijk        ]*valid[ijk-1      ] )+
                      fabs( beta_j[ijk        ]*valid[ijk-jStride] )+
                      fabs( beta_k[ijk        ]*valid[ijk-kStride] )+
                      fabs( beta_i[ijk+1      ]*valid[ijk+1      ] )+
                      fabs( beta_j[ijk+jStride]*valid[ijk+jStride] )+
                      fabs( beta_k[ijk+kStride]*valid[ijk+kStride] )
                      );

      // center of Gershgorin disc is the diagonal element...
      double    Aii = a*alpha[ijk] - b*h2inv*(
                                       beta_i[ijk        ]*( valid[ijk-1      ]-2.0 )+
                                       beta_j[ijk        ]*( valid[ijk-jStride]-2.0 )+
                                       beta_k[ijk        ]*( valid[ijk-kStride]-2.0 )+
                                       beta_i[ijk+1      ]*( valid[ijk+1      ]-2.0 )+
                                       beta_j[ijk+jStride]*( valid[ijk+jStride]-2.0 )+
                                       beta_k[ijk+kStride]*( valid[ijk+kStride]-2.0 ) 
                                     );

      #endif
                             Dinv[ijk] = 1.0/Aii;				// inverse of the diagonal Aii
                          //L1inv[ijk] = 1.0/(Aii+sumAbsAij);			// inverse of the L1 row norm... L1inv = ( D+D^{L1} )^{-1}
      // as suggested by eq 6.5 in Baker et al, "Multigrid smoothers for ultra-parallel computing: additional theory and discussion"...
      if(Aii>=1.5*sumAbsAij)L1inv[ijk] = 1.0/(Aii              ); 		//
                       else L1inv[ijk] = 1.0/(Aii+0.5*sumAbsAij);		// 
      double Di = (Aii + sumAbsAij)/Aii;if(Di>block_eigenvalue)block_eigenvalue=Di;	// upper limit to Gershgorin disc == bound on dominant eigenvalue
    }}}
    // if(block_eigenvalue>dominant_eigenvalue){dominant_eigenvalue = block_eigenvalue;}
    dominant_eigenvalue_atomic.update(block_eigenvalue);
  });
  level->cycles.blas1 += (uint64_t)(CycleTime()-_timeStart);

  double dominant_eigenvalue = dominant_eigenvalue_atomic.get();

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // Reduce the local estimates dominant eigenvalue to a global estimate
  #ifdef USE_MPI
  uint64_t _timeStartAllReduce = CycleTime();
  double send = dominant_eigenvalue;
  hclib::MPI_Allreduce(&send,&dominant_eigenvalue,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  uint64_t _timeEndAllReduce = CycleTime();
  level->cycles.collectives   += (uint64_t)(_timeEndAllReduce-_timeStartAllReduce);
  #endif
  if(level->my_rank==0){fprintf(stdout,"eigenvalue_max<%e\n",dominant_eigenvalue);}
  level->dominant_eigenvalue_of_DinvA = dominant_eigenvalue;


  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // exchange Dinv/L1inv/...
  exchange_boundary(level,VECTOR_DINV ,0); // must be 0(faces,edges,corners) for CA version
  exchange_boundary(level,VECTOR_L1INV,0);
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
}


//------------------------------------------------------------------------------------------------------------------------------
#ifdef  USE_GSRB
#define NUM_SMOOTHS      2 // RBRB
#include "operators/gsrb.c"
#elif   USE_CHEBY
#define NUM_SMOOTHS      1
#define CHEBYSHEV_DEGREE 4 // i.e. one degree-4 polynomial smoother
#include "operators/chebyshev.c"
#elif   USE_JACOBI
#define NUM_SMOOTHS      6
#include "operators/jacobi.c"
#elif   USE_L1JACOBI
#define NUM_SMOOTHS      6
#include "operators/jacobi.c"
#elif   USE_SYMGS
#define NUM_SMOOTHS      2
#include "operators/symgs.c"
#else
#error You must compile with either -DUSE_GSRB, -DUSE_CHEBY, -DUSE_JACOBI, -DUSE_L1JACOBI, or -DUSE_SYMGS
#endif
#include "operators/residual.c"
#include "operators/apply_op.c"
//------------------------------------------------------------------------------------------------------------------------------
#include "operators/blockCopy.c"
#include "operators/misc.c"
#include "operators/exchange_boundary.c"
#include "operators/boundary_conditions.c"
#include "operators/matmul.c"
#include "operators/restriction.c"
#include "operators/interpolation_pc.c"
#include "operators/interpolation_pl.c"
//#include "operators/interpolation_pq.c"
//------------------------------------------------------------------------------------------------------------------------------
void interpolation_vcycle(level_type * level_f, int id_f, double prescale_f, level_type *level_c, int id_c){interpolation_pc(level_f,id_f,prescale_f,level_c,id_c);}
void interpolation_fcycle(level_type * level_f, int id_f, double prescale_f, level_type *level_c, int id_c){interpolation_pl(level_f,id_f,prescale_f,level_c,id_c);}
//------------------------------------------------------------------------------------------------------------------------------
#include "operators/problem.p6.c"
//------------------------------------------------------------------------------------------------------------------------------
