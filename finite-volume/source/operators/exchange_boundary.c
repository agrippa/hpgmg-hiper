//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
// perform a (intra-level) ghost zone exchange
//  NOTE exchange_boundary() only exchanges the boundary.  
//  It will not enforce any boundary conditions
//  BC's are either the responsibility of a separate function or should be fused into the stencil
void exchange_boundary(level_type * level, int id, int justFaces){
  uint64_t _timeCommunicationStart = CycleTime();
  uint64_t _timeStart,_timeEnd;
  int buffer=0;
  int n;

  if(justFaces)justFaces=1;else justFaces=0;  // must be 0 or 1 in order to index into exchange_ghosts[]


  _timeStart = CycleTime();

#ifdef USE_UPCXX
  upcxx::barrier();
#elif USE_MPI
  int nMessages = level->exchange_ghosts[justFaces].num_recvs + level->exchange_ghosts[justFaces].num_sends;

  // loop through packed list of MPI receives and prepost Irecv's...
#ifdef USE_MPI_THREAD_MULTIPLE
#pragma omp parallel for schedule(dynamic,1)
#endif
  for(n=0;n<level->exchange_ghosts[justFaces].num_recvs;n++){
    MPI_Irecv(level->exchange_ghosts[justFaces].recv_buffers[n],
              level->exchange_ghosts[justFaces].recv_sizes[n],
              MPI_DOUBLE,
              level->exchange_ghosts[justFaces].recv_ranks[n],
              level->exchange_ghosts[justFaces].recv_ranks[n], // taged by sender's rank (i.e. recv tag is the tag of sending process)
              //0, // by construction, only one message should be received from each neighboring process
              MPI_COMM_WORLD,
              &level->exchange_ghosts[justFaces].requests[n]
    );
  }
#endif  
  _timeEnd = CycleTime();
  level->cycles.ghostZone_recv += (_timeEnd-_timeStart);

  // pack MPI send buffers...
  _timeStart = CycleTime();
#pragma omp parallel for if(level->exchange_ghosts[justFaces].num_blocks[0]>1) schedule(static,1)
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
      upcxx::async_copy(p1, p2, level->exchange_ghosts[justFaces].send_sizes[n]);
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
              level->my_rank, // tag with sender's rank (i.e. my rank)
              //0, // by construction, only one message should be sent to each neighboring process
              MPI_COMM_WORLD,
              &level->exchange_ghosts[justFaces].requests[n+level->exchange_ghosts[justFaces].num_recvs]
                                              // requests[0..num_recvs-1] were used by recvs.  So sends start at num_recvs
    ); 
  }
#endif
  _timeEnd = CycleTime();
  level->cycles.ghostZone_send += (_timeEnd-_timeStart);


  // exchange locally... try and hide within Isend latency... 
  _timeStart = CycleTime();
#pragma omp parallel for if(level->exchange_ghosts[justFaces].num_blocks[1]>1) schedule(static,1)
  for(buffer=0;buffer<level->exchange_ghosts[justFaces].num_blocks[1];buffer++){CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[1][buffer]);}
  _timeEnd = CycleTime();
  level->cycles.ghostZone_local += (_timeEnd-_timeStart);


  // wait for MPI to finish...
  _timeStart = CycleTime();

#ifdef USE_UPCXX
  async_copy_fence();
#elif USE_MPI 
  if(nMessages)MPI_Waitall(nMessages,level->exchange_ghosts[justFaces].requests,level->exchange_ghosts[justFaces].status);
  //if(level->exchange_ghosts[justFaces].num_sends)MPI_Waitall(level->exchange_ghosts[justFaces].num_sends,level->exchange_ghosts[justFaces].requests+level->exchange_ghosts[justFaces].num_recvs,level->exchange_ghosts[justFaces].status+level->exchange_ghosts[justFaces].num_recvs);
  //if(level->exchange_ghosts[justFaces].num_recvs)MPI_Waitall(level->exchange_ghosts[justFaces].num_recvs,level->exchange_ghosts[justFaces].requests,level->exchange_ghosts[justFaces].status); // wait just for recvs...
#endif
  _timeEnd = CycleTime();
  level->cycles.ghostZone_wait += (_timeEnd-_timeStart);


  // unpack MPI receive buffers 
  _timeStart = CycleTime();
#pragma omp parallel for if(level->exchange_ghosts[justFaces].num_blocks[2]>1) schedule(static,1)
  for(buffer=0;buffer<level->exchange_ghosts[justFaces].num_blocks[2];buffer++){CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[2][buffer]);}
  _timeEnd = CycleTime();
  level->cycles.ghostZone_unpack += (_timeEnd-_timeStart);

  // wait for sends to finish...
  //#ifdef USE_MPI
  //_timeStart = CycleTime();
  //if(level->exchange_ghosts[justFaces].num_sends)MPI_Waitall(level->exchange_ghosts[justFaces].num_sends,level->exchange_ghosts[justFaces].requests+level->exchange_ghosts[justFaces].num_recvs,level->exchange_ghosts[justFaces].status+level->exchange_ghosts[justFaces].num_recvs);
  //_timeEnd = CycleTime();
  //level->cycles.ghostZone_wait += (_timeEnd-_timeStart);
  //#endif
 
 
  level->cycles.ghostZone_total += (uint64_t)(CycleTime()-_timeCommunicationStart);
}
