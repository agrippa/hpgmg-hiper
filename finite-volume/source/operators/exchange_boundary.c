//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------

//Functions for AM 

static int iters = 0;
extern mg_type all_grids;

static volatile int allowSendingMsg = 1;

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
typedef void (*CB_INSIDE_FUNC)(double *, int, int, int, int, int, int);

void cb_copy(double *buf, int n, int srcid, int vid, int depth, int faces, int it) {

  uint64_t _timeCommunicationStart = CycleTime();
  uint64_t _timeStart,_timeEnd;

  int id = para_id;
  int justFaces = para_justFaces;
  level_type *level = para_level;

  int buffer;

  _timeStart = CycleTime();

  if (allowSendingMsg == 0) allowSendingMsg = 1;

/****
  PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[justFaces].num_blocks[2])
****/
  if (vid != id || depth != para_level->depth || faces != justFaces) {
#ifdef DEBUG
        printf("Warning Iter %d Proc %d In copy level %d faces %d  id %d to handle buf %p len %d from %d vid %d depth %d faces %d iter %d\n", 
          iters, MYTHREAD, para_level->depth, justFaces, id,             buf, n, srcid, vid, depth, faces, it); 
#endif
        if (all_grids.levels != NULL) {
#ifdef DEBUG
          printf("Warning depth %d all_grids %pi level %p %d\n", depth, &all_grids, all_grids.levels[depth], all_grids.levels[depth]->depth);
#endif
	  level = all_grids.levels[depth];
        }
        justFaces = faces;
        id = vid;
  }
  else {
#ifdef DEBUG
        printf("Iter %d Proc %d In copy level %d faces %d  id %d to handle buf %p len %d from %d iter %d\n", 
          iters, MYTHREAD, para_level->depth, justFaces, id,             buf, n, srcid, it);
#endif
  }

  int i;
  int nth = level->depth * 20 + id;
  for (i = 0; i < level->exchange_ghosts[justFaces].num_recvs; i++) {
     if (level->exchange_ghosts[justFaces].recv_ranks[i] == srcid) {
        if (level->exchange_ghosts[justFaces].flag_data[nth][i] != 0) {
	  printf("Wrong in Ping Handler Proc %d recv msg from %d for vid %d iter %d val %d\n", MYTHREAD, srcid, vid, it, 
	  level->exchange_ghosts[justFaces].flag_data[nth][i]);
	}
	else {
	  level->exchange_ghosts[justFaces].flag_data[nth][i] =1;
#ifdef DEBUG
          printf("Iter %d proc %d set single for nth %d pos %d\n", iters, MYTHREAD, nth, i);
#endif
	}
	break;
     }
  }
#ifdef DEBUG
  if (i >= level->exchange_ghosts[justFaces].num_recvs) {
	printf("Wrong again Proc %d not found %d from recv ranks level %d faces %d\n", MYTHREAD, srcid, level->depth, justFaces);
        for (int j = 0; j < level->exchange_ghosts[justFaces].num_recvs; j++) {
          printf("Level %d Proc %d recv pos %d is %d\n", level->depth, MYTHREAD, j, level->exchange_ghosts[justFaces].recv_ranks[j]);
        }
  }
  else {
       if (buf != level->exchange_ghosts[justFaces].recv_buffers[i]) 
	printf("Wrong buffer %p should be %p for Level %d Proc %d recv pos %d\n",
                      buf, level->exchange_ghosts[justFaces].recv_buffers[i], level->depth, MYTHREAD, i);
  }
#endif

  PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[justFaces].num_blocks[2])
  for(buffer=0;buffer<level->exchange_ghosts[justFaces].num_blocks[2];buffer++){
    // should make this more efficient, no need to go through all blocks
    // if (level->exchange_ghosts[justFaces].blocks[2][buffer].read.ptr == buf)
    if (level->exchange_ghosts[justFaces].blocks[2][buffer].read.box == -1-srcid)
      CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[2][buffer]);
  }
  _timeEnd = CycleTime();
  level->cycles.ghostZone_unpack += (_timeEnd-_timeStart);

}

void sendNbgrData(int rid, global_ptr<double> src, global_ptr<double> dest, int nelem) {

  int myid = gasnet_mynode(); 
  double * lsrc = (double *)src.raw_ptr();
  double * ldst = (double *)dest.raw_ptr();

#ifdef DEBUG
  cout << "Iter " << iters << " Proc " << gasnet_mynode() << " sending request to " << rid << " for " << nelem << 
    " elements from src " << src << " ( " << lsrc << ") to dest " << dest << " ( " << ldst << ")" <<  " AllowSending " << allowSendingMsg << endl;
#endif
  while (allowSendingMsg == 0) {
     gasnet_AMPoll();    
  }

#ifdef DEBUG
  printf("ALLOW now for proc %d depth %d id %d Iter %d\n", MYTHREAD, para_level->depth, para_id, iters);
#endif

  GASNET_Safe(gasnet_AMRequestLong4(rid, P2P_PING_LONGREQUEST, lsrc, nelem*sizeof(double), ldst, para_id, para_level->depth, para_justFaces, iters));

/*
  GASNET_Safe(gasnet_AMRequestMedium1(rid, P2P_PING_MEDREQUEST, lsrc, nelem*sizeof(double), myid));
*/

}

void syncNeighbor(int nbgr, int vid, int iter) {

#ifdef DEBUG
    cout << "Iter " << iter << " In sync " << upcxx::p2p_flag << " for proc " << gasnet_mynode() << " need " << nbgr << " vid " << vid << endl;
    GASNET_BLOCKUNTIL(upcxx::p2p_flag[vid] == nbgr);
    cout << "Iter " << iter << " P2p_FLAG is " << upcxx::p2p_flag[vid] << " for proc " << gasnet_mynode() << " vid " << vid << endl;
#else
    GASNET_BLOCKUNTIL(upcxx::p2p_flag[vid] == nbgr);
#endif
    upcxx::p2p_flag[vid] = 0;
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
  iters++;

#ifdef DEBUG
  if (level->exchange_ghosts[justFaces].num_sends+level->exchange_ghosts[justFaces].num_recvs == 0) 
	if (MYTHREAD != 0 ) allowSendingMsg = 1; 

  printf("EXCHANGE Proc %d for id %d level %d Iter %d justFaces %d nsend %d nrecv %d allow %d\n", MYTHREAD, id, level->depth, iters, justFaces,
         level->exchange_ghosts[justFaces].num_sends, level->exchange_ghosts[justFaces].num_recvs, allowSendingMsg);

  if (level->exchange_ghosts[justFaces].num_sends != level->exchange_ghosts[justFaces].num_recvs) {
	printf("Wrong msg send/recv number does not match send %d recvs %d for proc %d level %d id %d\n",
        level->exchange_ghosts[justFaces].num_sends, level->exchange_ghosts[justFaces].num_recvs, MYTHREAD, level->depth, id);
  }
#endif

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
  #else
    #ifdef USE_SUBCOMM
//      if (level->num_my_boxes == 0) MPI_Barrier(level->MPI_COMM_ALLREDUCE);
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

  int nth = level->depth * 20 + id;
  while (1) {
    int arrived = 0;
    for (int n = 0; n < level->exchange_ghosts[justFaces].num_recvs; n++) {
      if (level->exchange_ghosts[justFaces].flag_data[nth][n] == 1) arrived++;
    }
    if (arrived == level->exchange_ghosts[justFaces].num_recvs) break;
    upcxx::advance();
  }
  for (int n = 0; n < level->exchange_ghosts[justFaces].num_recvs; n++) {
    level->exchange_ghosts[justFaces].flag_data[nth][n] = 0;
  }

  _timeEnd = CycleTime();
  level->cycles.blas3 += (_timeEnd-_timeStart);

#ifdef USE_UPCXX
#ifndef UPCXX_P2P
  async_copy_fence();
#ifdef USE_SUBCOMM
  MPI_Barrier(level->MPI_COMM_ALLREDUCE);
#else
  upcxx::barrier();
#endif
#else
  syncNeighbor(level->exchange_ghosts[justFaces].num_sends, id, iters);
//  if (level->num_my_boxes == 0) MPI_Barrier(level->MPI_COMM_ALLREDUCE);
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
