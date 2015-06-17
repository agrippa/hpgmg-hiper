//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#ifndef LEVEL_H
#define LEVEL_H
//------------------------------------------------------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#ifdef USE_UPCXX
#include <upcxx.h>
using namespace upcxx;
#endif

#include "defines.h"

//------------------------------------------------------------------------------------------------------------------------------
#ifdef USE_MPI
#include <mpi.h>
#endif
//------------------------------------------------------------------------------------------------------------------------------
#define BC_PERIODIC  0
#define BC_DIRICHLET 1
//------------------------------------------------------------------------------------------------------------------------------
#ifndef BLOCKCOPY_TILE_I
#define BLOCKCOPY_TILE_I 10000
#else
#warning By overriding BLOCKCOPY_TILE_I, you are tiling in the unit stride.  I hope you know what you are doing.
#endif
#ifndef BLOCKCOPY_TILE_J
#define BLOCKCOPY_TILE_J 8
#endif
#ifndef BLOCKCOPY_TILE_K
#define BLOCKCOPY_TILE_K 8
#endif
//------------------------------------------------------------------------------------------------------------------------------

struct box_type;

typedef struct {
  int subtype;                  // e.g. used to calculate normal to domain for BC's
  struct {int i, j, k;}dim;     // dimensions of the block to copy
#ifdef USE_UPCXX
  struct {int box, i, j, k, jStride, kStride;double * __restrict__ ptr; global_ptr<box_type> boxgp;}read,write;
#else
  struct {int box, i, j, k, jStride, kStride;double * __restrict__ ptr;}read,write;
#endif
  // coordinates in the read grid to extract data, 
  // coordinates in the write grid to insert data
  // if read/write.box<0, then use write/read.ptr, otherwise use boxes[box].vectors[id]
  // Thus, you can do grid->grid, grid->buf, buf->grid, or buf->buf
} blockCopy_type;

#ifdef USE_UPCXX
extern shared_array< global_ptr<double>, 1 > upc_buf_info;
extern shared_array< global_ptr<int>, 1> upc_rflag_ptr;
#endif

//------------------------------------------------------------------------------------------------------------------------------
typedef struct {
    int                           num_recvs;	//   number of neighbors by type
    int                           num_sends;	//   number of neighbors by type
    int     * __restrict__       recv_ranks;	//   MPI rank of each neighbor...          recv_ranks[neighbor]
    int     * __restrict__       send_ranks;	//   MPI rank of each neighbor...          send_ranks[neighbor]
    int     * __restrict__       recv_sizes;	//   size of each MPI recv buffer...       recv_sizes[neighbor]
    int     * __restrict__       send_sizes;	//   size of each MPI send buffer...       send_sizes[neighbor]
#ifdef USE_UPCXX
    global_ptr<int>     rflag;                  //   message recv flag
    global_ptr<int>     *match_rflag;           //   address of neighbor's recv flag
    int                 *send_match_pos;        //   my position in receiver's recv list
    int              * __restrict__         sblock2, eblock2;  // start and end position in blocks[2] for each neighbor

    global_ptr<double>        *global_recv_buffers;    // allocate buffer in global address
    global_ptr<double>        *global_send_buffers;    // allocate buffer in global address
    global_ptr<double>        *global_match_buffers;   // record neighbor's recv buffer address

    event            *copy_e;                   //   events used for p2p synchronization      
    event            *data_e;                   //   events used for p2p synchronization 
#endif
    double ** __restrict__     recv_buffers;	//   MPI recv buffer for each neighbor...  recv_buffers[neighbor][ recv_sizes[neighbor] ]
    double ** __restrict__     send_buffers;	//   MPI send buffer for each neighbor...  send_buffers[neighbor][ send_sizes[neighbor] ]

  // shan: add one more block list for on node shared memory data access

    int                 allocated_blocks[4];	//   number of blocks allocated (not necessarily used) each list...
    int                       num_blocks[4];	//   number of blocks in each list...        num_blocks[pack,local,unpack]
    blockCopy_type *              blocks[4];	//   list of block copies...                     blocks[pack,local,unpack]
} communicator_type;


//------------------------------------------------------------------------------------------------------------------------------
typedef struct box_type {
  int                         global_box_id;	// used to inded into level->rank_of_box
  struct {int i, j, k;}low;			// global coordinates of the first (non-ghost) element of subdomain
  int                                   dim;	// dimension of this box's core (owned)
  int                                ghosts;	// ghost zone depth
  int                jStride,kStride,volume;	// useful for offsets
  int                            numVectors;	//
#ifdef USE_UPCXX
  global_ptr<global_ptr<double> >   vectors;    // allocate buffer in global address
  global_ptr<double>           vectors_base;    // allocate buffer in global address
#else
  double   ** __restrict__          vectors;	// vectors[c] = pointer to 3D array for vector c
  double    * __restrict__     vectors_base;    // pointer used for malloc/free.  vectors[c] are shifted from this for alignment
#endif
} box_type;


//------------------------------------------------------------------------------------------------------------------------------
typedef struct {
#ifdef USE_UPCXX
  int depth;
  team *subteam;
#endif
  double h;					// grid spacing at this level
  int active;					// I am an active process (I have work to do on this or subsequent levels)
  int num_ranks;				// total number of MPI ranks
  int my_rank;					// my MPI rank
  int box_dim;					// dimension of each cubical box (not counting ghost zones)
  int box_ghosts;				// ghost zone depth for each box
  int box_vectors;				// number of vectors stored in each box
  int tag;					// tag each level uniquely... FIX... replace with sub commuicator
  struct {int i, j, k;}boxes_in;		// total number of boxes in i,j,k across this level
  struct {int i, j, k;}dim;			// global dimensions at this level (NOTE: dim.i == boxes_in.i * box_dim)
  int * rank_of_box;				// 3D array containing rank of each box.  i-major ordering
  int    num_my_boxes;				//           number of boxes owned by this rank
#ifdef USE_UPCXX
  global_ptr<box_type> *addr_of_box;            // global address for all box, can be removed later
  global_ptr<box_type> my_boxes;                // global address
  box_type             **my_local_boxes;
#else
  box_type * my_boxes;				// pointer to array of boxes owned by this rank
#endif
  int       allocated_blocks;			//       number of blocks allocated by this rank (note, this represents a flattening of the box/cell hierarchy to facilitate threading)
  int          num_my_blocks;			//       number of blocks     owned by this rank (note, this represents a flattening of the box/cell hierarchy to facilitate threading)
  blockCopy_type * my_blocks;			// pointer to array of blocks owned by this rank (note, this represents a flattening of the box/cell hierarchy to facilitate threading)

  struct {
    int                type;                    // BC_PERIODIC or BC_DIRICHLET
    int    allocated_blocks[2];                 // number of blocks allocated (not necessarily used) for boundary conditions on this level for [0=all,1=justFaces]
    int          num_blocks[2];                 // number of blocks used for boundary conditions on this level for [0=all,1=justFaces]
    blockCopy_type * blocks[2];                 // pointer to array of blocks used for boundary conditions on this level for [0=all,1=justFaces]
  } boundary_condition;                         // boundary conditions on this level

  communicator_type exchange_ghosts[2];		// mini program that performs a neighbor ghost zone exchange for [0=all,1=justFaces]
  communicator_type restriction[4];		// mini program that performs restriction and agglomeration for [0=cell centered, 1=i-face, 2=j-face, 3-k-face]
  communicator_type interpolation;		// mini program that performs interpolation and dissemination...
#ifdef USE_MPI
  MPI_Comm MPI_COMM_ALLREDUCE;			// MPI sub communicator for just the ranks that have boxes on this level or any subsequent level... 
#endif
  double dominant_eigenvalue_of_DinvA;		// estimate on the dominate eigenvalue of D^{-1}A
  int alpha_is_zero;				// useful for determining Poisson... (a==0) && (alpha_is_zero)
  double    * __restrict__ RedBlack_FP[2];	// Red/Black Mask (i.e. 0.0 or 1.0) for even/odd planes (dim_with_ghosts^2).  

  int num_threads;
  int concurrent_boxes;
  int threads_per_box;

  // statistics information...
  uint64_t memory_allocated;			// 64b integer to track how much memory each process allocates for each level
  struct {
    uint64_t              smooth;
    uint64_t            apply_op;
    uint64_t            residual;
    uint64_t               blas1;
    uint64_t               blas3;
    uint64_t boundary_conditions;
    // Distributed Restriction
    uint64_t   restriction_total;
    uint64_t   restriction_pack;
    uint64_t   restriction_local;
    uint64_t   restriction_shm;
    uint64_t   restriction_unpack;
    uint64_t   restriction_recv;
    uint64_t   restriction_send;
    uint64_t   restriction_wait;
    // Distributed interpolation
    uint64_t interpolation_total;
    uint64_t interpolation_pack;
    uint64_t interpolation_local;
    uint64_t interpolation_shm;
    uint64_t interpolation_unpack;
    uint64_t interpolation_recv;
    uint64_t interpolation_send;
    uint64_t interpolation_wait;
    // Ghost Zone Exchanges...
    uint64_t     ghostZone_total;
    uint64_t     ghostZone_pack;
    uint64_t     ghostZone_local;
    uint64_t     ghostZone_shm;
    uint64_t     ghostZone_unpack;
    uint64_t     ghostZone_recv;
    uint64_t     ghostZone_send;
    uint64_t     ghostZone_wait;
    // Collectives...
    uint64_t   collectives;
    uint64_t         Total;
  }cycles;
  int Krylov_iterations;        // total number of bottom solver iterations
  int CAKrylov_formations_of_G; // i.e. [G,g] = [P,R]^T[P,R,rt]
  int vcycles_from_this_level;  // number of vcycles performed that were initiated from this level
} level_type;


//------------------------------------------------------------------------------------------------------------------------------
 int create_box(box_type *box, int numVectors, int dim, int ghosts);
void add_vectors_to_box(box_type *box, int numAdditionalVectors);
void destroy_box(box_type *box);
void create_level(level_type *level, int boxes_in_i, int box_dim, int box_ghosts, int box_vectors, int domain_boundary_condition, int my_rank, int num_ranks);
void destroy_level(level_type *level);
void reset_level_timers(level_type *level);
void   max_level_timers(level_type *level);
int qsortInt(const void *a, const void *b);
void append_block_to_list(blockCopy_type ** blocks, int *allocated_blocks, int *num_blocks,
                          int dim_i, int dim_j, int dim_k,
#ifdef USE_UPCXX
			  global_ptr<box_type> read_boxgp,
#endif
                          int  read_box, double*  read_ptr, int  read_i, int  read_j, int  read_k, int  read_jStride, int  read_kStride, int  read_scale,
#ifdef USE_UPCXX
                          global_ptr<box_type> write_boxgp,
#endif

                          int write_box, double* write_ptr, int write_i, int write_j, int write_k, int write_jStride, int write_kStride, int write_scale,
                          int my_blockcopy_tile_i, int my_blockcopy_tile_j, int my_blockcopy_tile_k,
                          int subtype
                         );
//------------------------------------------------------------------------------------------------------------------------------
#endif
