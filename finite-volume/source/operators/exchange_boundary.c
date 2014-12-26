//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------

//Functions for AM 

static int iters = 0;
extern mg_type *all_grids;
extern level_type *finest_level;
extern shared_array< global_ptr<mg_type>, 1> upc_grids;

void cb_unpack(double *buf, int srcid, int pos, int n, int vid, int depth, int faces) {

  uint64_t _timeCommunicationStart = CycleTime();
  uint64_t _timeStart,_timeEnd;

  int id = vid;
  int justFaces = faces;
  level_type *level;
  double *buf;

  int buffer;

  _timeStart = CycleTime();

  if (depth == 0) {
     level = finest_level;
  }
  else {
     level = all_grids->levels[depth];
  }

  int i;
  size_t nth = MAX_TLVG*(size_t)level->my_rank + MAX_LVG*faces + MAX_VG*depth + MAX_NBGS*vid;
  int *p = (int *) &upc_rflag[nth];
  for (i = 0; i < level->exchange_ghosts[justFaces].num_recvs; i++) {
     if (level->exchange_ghosts[justFaces].recv_ranks[i] == srcid) {
        if (p[i] != 0) {
	  printf("Wrong in Ping Handler Proc %d recv msg from %d for vid %d iter %d val %d\n", MYTHREAD, srcid, vid, it, upc_rflag[nth+i].get());
	}
	else {
          p[i] = 1; // upc_rflag[nth+i] =1;
	}
	break;
     }
  }
  assert(i == pos);
  assert(n > 0);
  buf = level->exchange_ghosts[justFaces].recv_buffers[pos];

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

  int bstart = level->exchange_ghosts[justFaces].sblock2[i];
  int bend   = level->exchange_ghosts[justFaces].sblock2[i+1];

  PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,bend-bstart)
  for(buffer=bstart;buffer<bend;buffer++){
    CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[2][buffer], buf, 0);
  }

  _timeEnd = CycleTime();
  level->cycles.ghostZone_unpack += (_timeEnd-_timeStart);

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
#ifdef USE_UPCXX
  event copy_e[27], data_e[27];
#endif

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
  int nshm = 0; 
  for(n=0;n<level->exchange_ghosts[justFaces].num_sends;n++){
    global_ptr<double> p1, p2;
    p1 = level->exchange_ghosts[justFaces].global_send_buffers[n];
    p2 = level->exchange_ghosts[justFaces].global_match_buffers[n];
#ifndef UPCXX_AM
    upcxx::async_copy(p1, p2, level->exchange_ghosts[justFaces].send_sizes[n]);
#else
    if (!is_memory_shared_with(level->exchange_ghosts[justFaces].send_ranks[n])) {
      upcxx::async_copy(p1, p2, level->exchange_ghosts[justFaces].send_sizes[n],&copy_e[n]);
    } else {
      int rid = level->exchange_ghosts[justFaces].send_ranks[n];
      int pos = level->exchange_ghosts[justFaces].send_match_pos[n];
      size_t nth = MAX_TLVG*(size_t)rid + MAX_LVG*justFaces + MAX_VG*level->depth + MAX_NBGS*id;
      int *p = (int *)&upc_rflag[nth+pos]; *p = 1; // upc_rflag[nth+pos] = 1;
      nshm++;
    }
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

  for(n=0;n<level->exchange_ghosts[justFaces].num_sends;n++){
    int rid = level->exchange_ghosts[justFaces].send_ranks[n];
    
    if (is_memory_shared_with(rid)) {
      // unpack buffer
    } else {
      int cnt = level->exchange_ghosts[justFaces].send_sizes[n];
      int pos = level->exchange_ghosts[justFaces].send_match_pos[n];
      async_after(rid, &copy_e[n], &data_e[n])(cb_unpack, level->my_rank, pos, cnt, id, level->depth, justFaces);
    }     
  }

  async_wait();

  size_t nth = MAX_TLVG*(size_t)level->my_rank + MAX_LVG*justFaces + MAX_VG*level->depth + MAX_NBGS*id;
  int *p = (int *) &upc_rflag[nth];
  while (1) {
    int arrived = 0;
    for (int n = 0; n < level->exchange_ghosts[justFaces].num_recvs; n++) {
      if (upc_rflag[nth + n] == 1) arrived++;
    }
    if (arrived == level->exchange_ghosts[justFaces].num_recvs) break;
    upcxx::advance();
  }
  for (int n = 0; n < level->exchange_ghosts[justFaces].num_recvs; n++) {
    p[n] = 0;  //upc_rflag[nth+n] = 0;
  }

  _timeEnd = CycleTime();
  level->cycles.blas3 += (_timeEnd-_timeStart);


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
 
  level->cycles.ghostZone_total += (uint64_t)(CycleTime()-_timeCommunicationStart);
}
