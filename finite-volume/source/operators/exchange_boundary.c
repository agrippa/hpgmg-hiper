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

  // fprintf(stderr, "cb_unpack: id=%d pos=%d on %d\n", id, pos, hclib::upcxx::myrank());

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

  // hclib::loop_domain_1d loop(bstart, bend);
  // hclib::future_t *fut = hclib::forasync1D_future(&loop, [=] (int buffer) {
  //         CopyBlock(level, id, &level->exchange_ghosts[justFaces].blocks[2][buffer]);
  //     }, (bend - bstart) <= 1, FORASYNC_MODE_FLAT, NULL);
  // fut->wait();

  // hclib::finish([&] {
  //     hclib::async([&] {
  //         fprintf(stderr, "HOWDY\n");
          // const unsigned long long start = hclib_current_time_ns();
          for(buffer=bstart;buffer<bend;buffer++){
              // CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[2][buffer], true);
              CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[2][buffer]);
          }
          // const unsigned long long elapsed = hclib_current_time_ns() - start;
          // fprintf(stderr, "HOWDY %llu ns\n", elapsed);
  //     });
  // });

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

  // must be 0 or 1 in order to index into exchange_ghosts[]
  if (justFaces) {
      justFaces=1;
  } else {
      justFaces=0;
  }

  // pack MPI send buffers...
  _timeStart = CycleTime();
  // PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[justFaces].num_blocks[0])
  hclib::future_t *fut0 = parallel_across_blocks(level, buffer, level->exchange_ghosts[justFaces].num_blocks[0],
          [&level, &id, &justFaces] (int buffer) {
      CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[0][buffer]);
  });
  fut0->wait();
  _timeEnd = CycleTime();
  level->cycles.ghostZone_pack += (_timeEnd-_timeStart);

  // loop through MPI send buffers and post Isend's...
  _timeStart = CycleTime();

  hclib::upcxx::remote_finish([&] {
      for(int n=0;n<level->exchange_ghosts[justFaces].num_sends;n++){
          hclib::upcxx::global_ptr<double> p1, p2;
          p1 = level->exchange_ghosts[justFaces].global_send_buffers[n];
          p2 = level->exchange_ghosts[justFaces].global_match_buffers[n];
          if (!hclib::upcxx::is_memory_shared_with(level->exchange_ghosts[justFaces].send_ranks[n])) {
              // Completion of async_copy triggers copy_e
              hclib::future_t *copy_e = hclib::upcxx::async_copy(p1, p2,
                  level->exchange_ghosts[justFaces].send_sizes[n]);

              int rid = level->exchange_ghosts[justFaces].send_ranks[n];
              int cnt = level->exchange_ghosts[justFaces].send_sizes[n];
              int pos = level->exchange_ghosts[justFaces].send_match_pos[n];
              // cb_unpack runs on remote node after copy completes, triggering data_e

              const int my_rank_copy = level->my_rank;
              const int depth_copy = level->depth;
              // fprintf(stderr, "%d sending cb_unpack to %d with id=%d pos=%d\n", hclib::upcxx::myrank(), rid, id, pos);
              hclib::upcxx::async_after(rid, copy_e, [=] {
                      cb_unpack(my_rank_copy, pos, cnt, id, depth_copy, justFaces);
                  });
          }
      }

      hclib::upcxx::advance();

      _timeEnd = CycleTime();
      level->cycles.ghostZone_send += (_timeEnd-_timeStart);

      // exchange locally... try and hide within Isend latency... 
      _timeStart = CycleTime();
      // PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[justFaces].num_blocks[3])
      hclib::future_t *fut1 = parallel_across_blocks(level, buffer, level->exchange_ghosts[justFaces].num_blocks[3],
              [&level, &id, &justFaces] (int buffer) {
          CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[3][buffer]);
      });

      fut1->wait();
      _timeEnd = CycleTime();
      level->cycles.ghostZone_shm += (_timeEnd-_timeStart);

      _timeStart = CycleTime();
      int nshm = 0;
      for(int n=0;n<level->exchange_ghosts[justFaces].num_sends;n++){
        if (hclib::upcxx::is_memory_shared_with(level->exchange_ghosts[justFaces].send_ranks[n])) {
          int rid = level->exchange_ghosts[justFaces].send_ranks[n];
          int pos = level->exchange_ghosts[justFaces].send_match_pos[n];
          size_t nth = MAX_NBGS * id;
          int *p = (int *)level->exchange_ghosts[justFaces].match_rflag[n]; *(p+nth+pos) = 1; // upc_rflag[nth+pos] = 1;
          nshm++;
        }
      }
      _timeEnd = CycleTime();
      level->cycles.ghostZone_shm += (_timeEnd-_timeStart);

      _timeStart = CycleTime();
      // PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[justFaces].num_blocks[1])
      hclib::future_t *fut2 = parallel_across_blocks(level, buffer, level->exchange_ghosts[justFaces].num_blocks[1],
              [&level, &id, &justFaces] (int buffer) {
          CopyBlock(level,id,&level->exchange_ghosts[justFaces].blocks[1][buffer]);
      });
      fut2->wait();
      _timeEnd = CycleTime();
      level->cycles.ghostZone_local += (_timeEnd-_timeStart);

      _timeStart = CycleTime();
      // Wait for incoming messages to be received, signalled by cb_unpack
      if (level->exchange_ghosts[justFaces].num_recvs > 0) {
          size_t nth = MAX_NBGS * id;
          int *p = (int *) level->exchange_ghosts[justFaces].rflag;
          // fprintf(stderr, "%d : spinning...\n", hclib::upcxx::myrank());
          while (1) {
              int arrived = 0;
              for (int n = 0; n < level->exchange_ghosts[justFaces].num_recvs; n++) {
                  if (level->exchange_ghosts[justFaces].rflag[nth + n] == 1) {
                      arrived++;
                  }
              }
              if (arrived == level->exchange_ghosts[justFaces].num_recvs) break;
              hclib::upcxx::advance();
          }
          // fprintf(stderr, "%d : done spinning...\n", hclib::upcxx::myrank());
          for (int n = 0; n < level->exchange_ghosts[justFaces].num_recvs; n++) {
              p[nth+n] = 0;  //upc_rflag[nth+n] = 0;
          }
      }
  });
  // fprintf(stderr, "%d : done waiting...\n", hclib::upcxx::myrank());

  _timeEnd = CycleTime();
  level->cycles.ghostZone_wait += (_timeEnd-_timeStart);
 
  level->cycles.ghostZone_total += (uint64_t)(CycleTime()-_timeCommunicationStart);
}
