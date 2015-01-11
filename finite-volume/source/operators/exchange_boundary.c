//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------

//Functions for asyncafter 

#ifdef USE_UPCXX
extern mg_type *all_grids;
void cb_unpack(int srcid, int pos, int n, int id, int depth, int justFaces) {

  uint64_t _timeCommunicationStart = CycleTime();
  uint64_t _timeStart,_timeEnd;

  level_type *level = all_grids->levels[depth];
  int buffer;

  _timeStart = CycleTime();

  int i;
  size_t nth = id * MAX_NBGS;
  int *p = (int *) level->exchange_ghosts[justFaces].rflag;
  if (p[nth+pos] != 0) {
    printf("Wrong in Ping Handler Proc %d recv msg from %d for id %d val %d\n", MYTHREAD, srcid, id, p[nth+pos]);
  }
  else {
    p[nth+pos] = 1; // upc_rflag[nth+i] =1;
  }

  int bstart = level->exchange_ghosts[justFaces].sblock2[pos];
  int bend   = level->exchange_ghosts[justFaces].sblock2[pos+1];

  //  PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,bend-bstart)
  for(buffer=bstart;buffer<bend;buffer++){
    CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[2][buffer]);
  }

  _timeEnd = CycleTime();
  level->cycles.ghostZone_unpack += (_timeEnd-_timeStart);

}

#endif // USE_UPCXX

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

  _timeStart = CycleTime();
#ifdef USE_UPCXX
#ifndef UPCXX_AM
#ifdef USE_SUBCOMM
  MPI_Barrier(level->MPI_COMM_ALLREDUCE);
#else
  upcxx::barrier();
#endif
#endif
#endif  
  _timeEnd = CycleTime();
  level->cycles.ghostZone_wait += (_timeEnd-_timeStart);

  // exchange locally... try and hide within Isend latency... 
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[justFaces].num_blocks[1])
  for(buffer=0;buffer<level->exchange_ghosts[justFaces].num_blocks[1];buffer++){CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[1][buffer]);}
  _timeEnd = CycleTime();
  level->cycles.ghostZone_local += (_timeEnd-_timeStart);

  // pack MPI send buffers...
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[justFaces].num_blocks[0])
  for(buffer=0;buffer<level->exchange_ghosts[justFaces].num_blocks[0];buffer++){CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[0][buffer]);}
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
      event* copy_e = &level->exchange_ghosts[justFaces].copy_e[n];
      upcxx::async_copy(p1, p2, level->exchange_ghosts[justFaces].send_sizes[n],copy_e);
    } else {
      int rid = level->exchange_ghosts[justFaces].send_ranks[n];
      int pos = level->exchange_ghosts[justFaces].send_match_pos[n];
      size_t nth = MAX_NBGS * id;
      int *p = (int *)level->exchange_ghosts[justFaces].match_rflag[n]; *(p+nth+pos) = 1; // upc_rflag[nth+pos] = 1;
      nshm++;
    }
#endif
  }
#endif
  _timeEnd = CycleTime();
  level->cycles.ghostZone_send += (_timeEnd-_timeStart);

  // wait for MPI to finish...
  _timeStart = CycleTime();

#ifdef USE_UPCXX
#ifdef UPCXX_AM

  for(n=0;n<level->exchange_ghosts[justFaces].num_sends;n++){
    int rid = level->exchange_ghosts[justFaces].send_ranks[n];
    
    if (!is_memory_shared_with(rid)) {
      int cnt = level->exchange_ghosts[justFaces].send_sizes[n];
      int pos = level->exchange_ghosts[justFaces].send_match_pos[n];
      event* copy_e = &level->exchange_ghosts[justFaces].copy_e[n];
      event* data_e = &level->exchange_ghosts[justFaces].data_e[n];
      async_after(rid, copy_e, data_e)(cb_unpack, level->my_rank, pos, cnt, id, level->depth, justFaces);
    }     
  }

  async_wait();

  if (level->exchange_ghosts[justFaces].num_recvs > 0) {
  size_t nth = MAX_NBGS * id;
  int *p = (int *) level->exchange_ghosts[justFaces].rflag;
  while (1) {
    int arrived = 0;
    for (int n = 0; n < level->exchange_ghosts[justFaces].num_recvs; n++) {
      if (level->exchange_ghosts[justFaces].rflag[nth + n] == 1) arrived++;
    }
    if (arrived == level->exchange_ghosts[justFaces].num_recvs) break;
    upcxx::advance();
  }
  for (int n = 0; n < level->exchange_ghosts[justFaces].num_recvs; n++) {
    p[nth+n] = 0;  //upc_rflag[nth+n] = 0;
  }

  }

#else  // UPCXX_AM

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
