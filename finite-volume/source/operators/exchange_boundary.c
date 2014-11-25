//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------

//Functions for AM 

#define GASNET_Safe(fncall) do {                                     \
  int _retval;                                                     \
    if ((_retval = fncall) != GASNET_OK) {                           \
      fprintf(stderr, "ERROR calling: %s\n"                          \
                   " at: %s:%i\n"                                    \
                   " error: %s (%s)\n",                              \
              #fncall, __FILE__, __LINE__,                           \
              gasnet_ErrorName(_retval), gasnet_ErrorDesc(_retval)); \
      fflush(stderr);                                                \
      gasnet_exit(_retval);                                          \
    }                                                                \
  } while(0)

// can use a function to pass these parameters
static int para_id, para_justFaces;
static level_type *para_level;
typedef void (*CB_INSIDE_FUNC)(double *, int);

void cb_copy(double *buf, int n) {

  uint64_t _timeCommunicationStart = CycleTime();
  uint64_t _timeStart,_timeEnd;

  int id = para_id;
  int justFaces = para_justFaces;
  level_type *level = para_level;

  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[justFaces].num_blocks[2])
  for(buffer=0;buffer<level->exchange_ghosts[justFaces].num_blocks[2];buffer++){
    // should make this more efficient, no need to go through all blocks
    if (level->exchange_ghosts[justFaces].blocks[2][buffer].read_ptr == buf)
      CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[2][buffer]);
  }
  _timeEnd = CycleTime();
  level->cycles.ghostZone_unpack += (_timeEnd-_timeStart);

}

void sendNbgrData(int rid, global_ptr<double> src, global_ptr<double> dest, int nelem) {

  int myid = gasnet_mynode(); 
  double * lsrc = (double *)src.raw_ptr();
  double * ldst = (double *)dest.raw_ptr();
  cout << "Proc " << gasnet_mynode() << " sending request to " << rid << " for " << nelem << 
    " elements from src " << src << " ( " << lsrc << ") to dest " << dest << " ( " << ldst << ")" << endl;
  
  GASNET_Safe(gasnet_AMRequestLong1(rid, P2P_PING_LONGREQUEST, lsrc, nelem*sizeof(double), ldst, myid));

/*
  GASNET_Safe(gasnet_AMRequestMedium1(rid, P2P_PING_MEDREQUEST, lsrc, nelem*sizeof(double), myid));
*/

}

void syncNeighbor(int nbgr) {

    GASNET_BLOCKUNTIL(upcxx::p2p_flag == nbgr);
    cout << "P2p_FLAG is " << upcxx::p2p_flag << " for proc " << gasnet_mynode() << endl;
    upcxx::p2p_flag = 0;
}

// perform a (intra-level) ghost zone exchange
//  NOTE exchange_boundary() only exchanges the boundary.  
//  It will not enforce any boundary conditions
//  BC's are either the responsibility of a separate function or should be fused into the stencil
void exchange_boundary(level_type * level, int id, int justFaces){
  uint64_t _timeCommunicationStart = CycleTime();
  uint64_t _timeStart,_timeEnd;
  int my_tag = (level->tag<<4) | justFaces;
  int buffer=0;
  int n;

  if(justFaces)justFaces=1;else justFaces=0;  // must be 0 or 1 in order to index into exchange_ghosts[]

#ifdef UPCXX_P2P
  // For cb_copy function: this can be set as AM parameters
  para_id = id;
  para_justFaces = justFaces;
  para_level = level;
  setCBFunc(cb_copy); // could set earlier in initialization, for convenience now
#endif

  _timeStart = CycleTime();

#ifdef USE_UPCXX
#ifndef UPCXX_P2P
#ifdef USE_SUBCOMM
  MPI_Barrier(level->MPI_COMM_ALLREDUCE);
#else
  upcxx::barrier();
#endif
#endif

#elif USE_MPI
  int nMessages = level->exchange_ghosts[justFaces].num_recvs + level->exchange_ghosts[justFaces].num_sends;
  MPI_Request *recv_requests = level->exchange_ghosts[justFaces].requests;
  MPI_Request *send_requests = level->exchange_ghosts[justFaces].requests + level->exchange_ghosts[justFaces].num_recvs;

  // loop through packed list of MPI receives and prepost Irecv's...
#ifdef USE_MPI_THREAD_MULTIPLE
#pragma omp parallel for schedule(dynamic,1)
#endif
  for(n=0;n<level->exchange_ghosts[justFaces].num_recvs;n++){
    MPI_Irecv(level->exchange_ghosts[justFaces].recv_buffers[n],
              level->exchange_ghosts[justFaces].recv_sizes[n],
              MPI_DOUBLE,
              level->exchange_ghosts[justFaces].recv_ranks[n],
              my_tag,
              MPI_COMM_WORLD,
              &recv_requests[n]
    );
  }
#endif  
  _timeEnd = CycleTime();
  level->cycles.ghostZone_recv += (_timeEnd-_timeStart);

  // pack MPI send buffers...
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[justFaces].num_blocks[0])
  for(buffer=0;buffer<level->exchange_ghosts[justFaces].num_blocks[0];buffer++){CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[0][buffer]);}
  _timeEnd = CycleTime();
  level->cycles.ghostZone_pack += (_timeEnd-_timeStart);

 
  // loop through MPI send buffers and post Isend's...
  _timeStart = CycleTime();

#ifdef USE_UPCXX
  for(n=0;n<level->exchange_ghosts[justFaces].num_sends;n++){
    global_ptr<double> p1, p2;
    p1 = level->exchange_ghosts[justFaces].global_send_buffers[n];
    p2 = level->exchange_ghosts[justFaces].global_match_buffers[n];
#ifndef UPCXX_P2P
    upcxx::async_copy(p1, p2, level->exchange_ghosts[justFaces].send_sizes[n]);
#else
    sendNbgrData(level->exchange_ghosts[justFaces].send_ranks[n], 
		 p1, p2, level->exchange_ghosts[justFaces].send_sizes[n]);
#endif
  }

#elif USE_MPI
#ifdef USE_MPI_THREAD_MULTIPLE
#pragma omp parallel for schedule(dynamic,1)
#endif
  for(n=0;n<level->exchange_ghosts[justFaces].num_sends;n++){
    MPI_Isend(level->exchange_ghosts[justFaces].send_buffers[n],
              level->exchange_ghosts[justFaces].send_sizes[n],
              MPI_DOUBLE,
              level->exchange_ghosts[justFaces].send_ranks[n],
              my_tag,
              MPI_COMM_WORLD,
              &send_requests[n]
    ); 
  }
#endif
  _timeEnd = CycleTime();
  level->cycles.ghostZone_send += (_timeEnd-_timeStart);


  // exchange locally... try and hide within Isend latency... 
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[justFaces].num_blocks[1])
  for(buffer=0;buffer<level->exchange_ghosts[justFaces].num_blocks[1];buffer++){CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[1][buffer]);}
  _timeEnd = CycleTime();
  level->cycles.ghostZone_local += (_timeEnd-_timeStart);


  // wait for MPI to finish...
  _timeStart = CycleTime();

#ifdef USE_UPCXX
#ifndef UPCXX_P2P
  async_copy_fence();
#ifdef USE_SUBCOMM
  MPI_Barrier(level->MPI_COMM_ALLREDUCE);
#else
  upcxx::barrier();
#endif
#else
  syncNeighbor(level->exchange_ghosts[justFaces].num_sends);
#endif

#elif USE_MPI 
  if(nMessages)MPI_Waitall(nMessages,level->exchange_ghosts[justFaces].requests,level->exchange_ghosts[justFaces].status);
#endif
  _timeEnd = CycleTime();
  level->cycles.ghostZone_wait += (_timeEnd-_timeStart);


#ifndef UPCXX_P2P
  // unpack MPI receive buffers 
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[justFaces].num_blocks[2])
  for(buffer=0;buffer<level->exchange_ghosts[justFaces].num_blocks[2];buffer++){CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[2][buffer]);}
  _timeEnd = CycleTime();
  level->cycles.ghostZone_unpack += (_timeEnd-_timeStart);
#endif
 
  level->cycles.ghostZone_total += (uint64_t)(CycleTime()-_timeCommunicationStart);
}
