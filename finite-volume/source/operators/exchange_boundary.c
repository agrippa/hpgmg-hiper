//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------

//Functions for AM 

static int iters = 0;
extern mg_type all_grids;
extern level_type *finest_level;

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

typedef void (*CB_INSIDE_FUNC)(double *, int, int, int, int, int, int);

void cb_copy(double *buf, int n, int srcid, int vid, int depth, int faces, int it) {

  uint64_t _timeCommunicationStart = CycleTime();
  uint64_t _timeStart,_timeEnd;

  int id = vid;
  int justFaces = faces;
  level_type *level;

  int buffer;

  _timeStart = CycleTime();

  if (depth == 0) {
     level = finest_level;
  }
  else {
     level = all_grids.levels[depth];
  }

  int i;
  int nth = depth * 20 + id;
  for (i = 0; i < level->exchange_ghosts[justFaces].num_recvs; i++) {
     if (level->exchange_ghosts[justFaces].recv_ranks[i] == srcid) {
        if (level->exchange_ghosts[justFaces].rflag[vid][i] != 0) {
	  printf("Wrong in Ping Handler Proc %d recv msg from %d for vid %d iter %d val %d\n", MYTHREAD, srcid, vid, it, 
	  level->exchange_ghosts[justFaces].rflag[vid][i]);
	}
	else {
	  level->exchange_ghosts[justFaces].rflag[vid][i] =1;
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

  if (n > 0) {
  int msize = gasnet_AMMaxMedium();
  int bstart = level->exchange_ghosts[justFaces].sblock2[i];
  int bend   = level->exchange_ghosts[justFaces].sblock2[i+1];

  if (n < msize) { // medium AM 
    PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer, bend-bstart)
    for(buffer=bstart;buffer<bend;buffer++){
      // if (level->exchange_ghosts[justFaces].blocks[2][buffer].read.ptr == buf)
      //if (level->exchange_ghosts[justFaces].blocks[2][buffer].read.box != -1-srcid) {
      //            level->exchange_ghosts[justFaces].blocks[2][buffer].read.box * (-1) - 1);
      //}
      CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[2][buffer], buf, 1);
    }
  }
  else { // long AM
    PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,bend-bstart)
    for(buffer=bstart;buffer<bend;buffer++){
      // if (level->exchange_ghosts[justFaces].blocks[2][buffer].read.ptr == buf)
      //if (level->exchange_ghosts[justFaces].blocks[2][buffer].read.box != -1-srcid) {
      //    printf("Error srcid long in proc %d should be %d actually be %d : %d\n", MYTHREAD, level->exchange_ghosts[justFaces].recv_ranks[i], srcid,
      //            level->exchange_ghosts[justFaces].blocks[2][buffer].read.box * (-1) - 1);
      //}
      CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[2][buffer], buf, 0);
    }
  }
  } // n > 0

  _timeEnd = CycleTime();
  level->cycles.ghostZone_unpack += (_timeEnd-_timeStart);

}

void sendNbgrData(int rid, global_ptr<double> src, global_ptr<double> dest, int nelem, int vid, int depth, int faces, int iters) {

  int myid = gasnet_mynode(); 
  double * lsrc = (double *)src.raw_ptr();
  double * ldst = (double *)dest.raw_ptr();
  int msize = gasnet_AMMaxMedium();

  if (nelem * sizeof(double) < msize) {
     // using mediumAM
    GASNET_Safe(gasnet_AMRequestMedium4(rid, P2P_PING_MEDREQUEST, lsrc, nelem*sizeof(double), vid, depth, faces, iters));
  }
  else {
    GASNET_Safe(gasnet_AMRequestLongAsync4(rid, P2P_PING_LONGREQUEST, lsrc, nelem*sizeof(double), ldst, vid, depth, faces, iters));
  }

}

void syncNeighbor(int nbgr, int depth, int vid, int iter) {

    GASNET_BLOCKUNTIL(upcxx::p2p_flag[depth][vid] == nbgr);
    upcxx::p2p_flag[depth][vid] = 0;
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
  printf("EXCHANGE Proc %d for id %d level %d Iter %d justFaces %d nsend %d nrecv %d \n", MYTHREAD, id, level->depth, iters, justFaces,
         level->exchange_ghosts[justFaces].num_sends, level->exchange_ghosts[justFaces].num_recvs);

  if (level->exchange_ghosts[justFaces].num_sends != level->exchange_ghosts[justFaces].num_recvs) {
	printf("Wrong msg send/recv number does not match send %d recvs %d for proc %d level %d id %d\n",
        level->exchange_ghosts[justFaces].num_sends, level->exchange_ghosts[justFaces].num_recvs, MYTHREAD, level->depth, id);
  }
#endif

  _timeStart = CycleTime();

#ifdef USE_UPCXX
  #ifndef UPCXX_AM
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

#endif  
  _timeEnd = CycleTime();
  level->cycles.ghostZone_recv += (_timeEnd-_timeStart);

  // exchange locally... try and hide within Isend latency... 
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[justFaces].num_blocks[1])
  for(buffer=0;buffer<level->exchange_ghosts[justFaces].num_blocks[1];buffer++){CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[1][buffer], NULL, 3);}
  _timeEnd = CycleTime();
  level->cycles.ghostZone_local += (_timeEnd-_timeStart);

  // pack MPI send buffers...
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[justFaces].num_blocks[0])
  for(buffer=0;buffer<level->exchange_ghosts[justFaces].num_blocks[0];buffer++){CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[0][buffer], NULL, 2);}
  _timeEnd = CycleTime();
  level->cycles.ghostZone_pack += (_timeEnd-_timeStart);

 
  // loop through MPI send buffers and post Isend's...
  _timeStart = CycleTime();

#ifdef USE_UPCXX
  for(n=0;n<level->exchange_ghosts[justFaces].num_sends;n++){
    global_ptr<double> p1, p2;
    p1 = level->exchange_ghosts[justFaces].global_send_buffers[n];
    p2 = level->exchange_ghosts[justFaces].global_match_buffers[n];
#ifndef UPCXX_AM
    upcxx::async_copy(p1, p2, level->exchange_ghosts[justFaces].send_sizes[n]);
#else
    sendNbgrData(level->exchange_ghosts[justFaces].send_ranks[n], 
		 p1, p2, level->exchange_ghosts[justFaces].send_sizes[n], id, level->depth, justFaces, iters);
#endif
  }

#endif
  _timeEnd = CycleTime();
  level->cycles.ghostZone_send += (_timeEnd-_timeStart);


/****  move temporarily for testing
  // exchange locally... try and hide within Isend latency... 
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[justFaces].num_blocks[1])
  for(buffer=0;buffer<level->exchange_ghosts[justFaces].num_blocks[1];buffer++){CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[1][buffer], NULL, 3);}
  _timeEnd = CycleTime();
  level->cycles.ghostZone_local += (_timeEnd-_timeStart);
****/

  // wait for MPI to finish...
  _timeStart = CycleTime();

#ifdef USE_UPCXX
#ifdef UPCXX_AM
  while (1) {
    int arrived = 0;
    for (int n = 0; n < level->exchange_ghosts[justFaces].num_recvs; n++) {
      if (level->exchange_ghosts[justFaces].rflag[id][n] == 1) arrived++;
    }
    if (arrived == level->exchange_ghosts[justFaces].num_recvs) break;
    upcxx::advance();
  }
  for (int n = 0; n < level->exchange_ghosts[justFaces].num_recvs; n++) {
    level->exchange_ghosts[justFaces].rflag[id][n] = 0;
  }

  _timeEnd = CycleTime();
  level->cycles.blas3 += (_timeEnd-_timeStart);

  syncNeighbor(level->exchange_ghosts[justFaces].num_sends, level->depth, id, iters);
//  if (level->num_my_boxes == 0) MPI_Barrier(level->MPI_COMM_ALLREDUCE);

#else

  async_copy_fence();
#ifdef USE_SUBCOMM
  MPI_Barrier(level->MPI_COMM_ALLREDUCE);
#else
  upcxx::barrier();
#endif
#endif
#endif
  _timeEnd = CycleTime();
  level->cycles.ghostZone_wait += (_timeEnd-_timeStart);


#ifndef UPCXX_AM
  // unpack MPI receive buffers 
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[justFaces].num_blocks[2])
  for(buffer=0;buffer<level->exchange_ghosts[justFaces].num_blocks[2];buffer++){CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[2][buffer], NULL, 4);}
  _timeEnd = CycleTime();
  level->cycles.ghostZone_unpack += (_timeEnd-_timeStart);
#endif
 
  level->cycles.ghostZone_total += (uint64_t)(CycleTime()-_timeCommunicationStart);
}
