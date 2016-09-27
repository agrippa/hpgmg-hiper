//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
// shan: I noticed that if we use UPCXX_AM for this file, the first iteration results are not correct.
// However, after the first iteration, everything is ok. The reason is not clearly understood yet.
// Two solutions now: 1. use a barrier at beginning 2. diable unpacking inside handler
// This is method 2.

void cb_unpack_int(int srcid, int pos, int depth_f, int id_f, double prescale_f) {

  uint64_t _timeCommunicationStart = CycleTime();
  uint64_t _timeStart,_timeEnd;

  level_type *level_f;
  int buffer;

  _timeStart = CycleTime();

  level_f = all_grids->levels[depth_f];

  size_t nth = MAX_NBGS*id_f;  nth = 0;
  int *p = (int *) level_f->interpolation.rflag;
  if (p[nth+pos] != 0) {
    printf("Wrong in Ping Res Handler Proc %d recv msg from %d for id_f %d val %d\n", MYTHREAD, srcid, id_f, p[nth+pos]);
  }
  else {
    p[nth+pos] =1;
  }

  int bstart = level_f->interpolation.sblock2[pos];
  int bend   = level_f->interpolation.sblock2[pos+1];

  for(buffer=bstart;buffer<bend;buffer++){
    IncrementBlock(level_f,id_f,prescale_f,&level_f->interpolation.blocks[2][buffer]);
  }

  _timeEnd = CycleTime();
  level_f->cycles.interpolation_unpack += (_timeEnd-_timeStart);

}

static inline void InterpolateBlock_PC(level_type *level_f, int id_f, double prescale_f, level_type *level_c, int id_c, blockCopy_type *block){
  // interpolate 3D array from read_i,j,k of read[] to write_i,j,k in write[]
  int   dim_i       = block->dim.i<<1; // calculate the dimensions of the resultant fine block
  int   dim_j       = block->dim.j<<1;
  int   dim_k       = block->dim.k<<1;

  int  read_i       = block->read.i;
  int  read_j       = block->read.j;
  int  read_k       = block->read.k;
  int  read_jStride = block->read.jStride;
  int  read_kStride = block->read.kStride;

  int write_i       = block->write.i;
  int write_j       = block->write.j;
  int write_k       = block->write.k;
  int write_jStride = block->write.jStride;
  int write_kStride = block->write.kStride;

  double * __restrict__  read = block->read.ptr;
  double * __restrict__ write = block->write.ptr;
  if(block->read.box >=0){
#ifdef USE_UPCXX
    box_type *lbox = (box_type *) block->read.boxgp;
    hclib::upcxx::global_ptr<double> gp = lbox->vectors[id_c] + lbox->ghosts*(1+lbox->jStride+lbox->kStride); 
    read = (double *)gp;
    read_jStride = lbox->jStride;
    read_kStride = lbox->kStride;
#else   // USE_UPCXX
     read = level_c->my_boxes[ block->read.box].vectors[id_c] + level_c->my_boxes[ block->read.box].ghosts*(1+level_c->my_boxes[ block->read.box].jStride+level_c->my_boxes[ block->read.box].kStride);
     read_jStride = level_c->my_boxes[block->read.box ].jStride;
     read_kStride = level_c->my_boxes[block->read.box ].kStride;
#endif
  }
  if(block->write.box>=0){
#ifdef USE_UPCXX
    box_type *lbox = (box_type *) block->write.boxgp;
    hclib::upcxx::global_ptr<double> gp = lbox->vectors[id_f] + lbox->ghosts*(1+lbox->jStride+lbox->kStride); 
    write = (double *)gp;
    write_jStride = lbox->jStride;
    write_kStride = lbox->kStride;
#else  // USE_UPCXX
    write = level_f->my_boxes[block->write.box].vectors[id_f] + level_f->my_boxes[block->write.box].ghosts*(1+level_f->my_boxes[block->write.box].jStride+level_f->my_boxes[block->write.box].kStride);
    write_jStride = level_f->my_boxes[block->write.box].jStride;
    write_kStride = level_f->my_boxes[block->write.box].kStride;
#endif
  }
 
 
  int i,j,k;
  for(k=0;k<dim_k;k++){
  for(j=0;j<dim_j;j++){
  for(i=0;i<dim_i;i++){
    int write_ijk = ((i   )+write_i) + (((j   )+write_j)*write_jStride) + (((k   )+write_k)*write_kStride);
    int  read_ijk = ((i>>1)+ read_i) + (((j>>1)+ read_j)* read_jStride) + (((k>>1)+ read_k)* read_kStride);
    write[write_ijk] = prescale_f*write[write_ijk] + read[read_ijk]; // CAREFUL !!!  you must guarantee you zero'd the MPI buffers(write[]) and destination boxes at some point to avoid 0.0*NaN or 0.0*inf
  }}}

}


//------------------------------------------------------------------------------------------------------------------------------
// perform a (inter-level) piecewise constant interpolation
void interpolation_pc(level_type * level_f, int id_f, double prescale_f, level_type *level_c, int id_c){
  uint64_t _timeCommunicationStart = CycleTime();
  uint64_t _timeStart,_timeEnd;
  int my_tag = (level_f->tag<<4) | 0x6;
  int buffer=0;
  int n;

  // perform local interpolation... try and hide within Isend latency... 
  _timeStart = CycleTime();
  // PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_c->interpolation.num_blocks[3])
  hclib::future_t *fut0 = parallel_across_blocks(level_f, buffer, level_c->interpolation.num_blocks[3],
          [&level_f, &id_f, &prescale_f, &level_c, &id_c] (int buffer) {
    InterpolateBlock_PC(level_f,id_f,prescale_f,level_c,id_c,
        &level_c->interpolation.blocks[3][buffer]);
  });
  fut0->wait();
  _timeEnd = CycleTime();
  level_f->cycles.interpolation_shm += (_timeEnd-_timeStart);

  // pack MPI send buffers...
  _timeStart = CycleTime();
  // PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_c->interpolation.num_blocks[0])
  hclib::future_t *fut1 = parallel_across_blocks(level_f, buffer, level_c->interpolation.num_blocks[0],
          [&level_f, &id_f, &level_c, &id_c] (int buffer) {
    InterpolateBlock_PC(level_f,id_f,0.0,level_c,id_c,
        &level_c->interpolation.blocks[0][buffer]);
  });
  fut1->wait();
  _timeEnd = CycleTime();
  level_f->cycles.interpolation_pack += (_timeEnd-_timeStart);

  hclib::future_t **events = (hclib::future_t **)malloc(
          level_c->interpolation.num_sends * sizeof(hclib::future_t *));
  memset(events, 0x00,
          level_c->interpolation.num_sends * sizeof(hclib::future_t *));
 
  // loop through MPI send buffers and post Isend's...
  _timeStart = CycleTime();
#ifdef USE_UPCXX
  int nshm = 0;
  for(n=0;n<level_c->interpolation.num_sends;n++){
    hclib::upcxx::global_ptr<double> p1, p2;
    p1 = level_c->interpolation.global_send_buffers[n];
    p2 = level_c->interpolation.global_match_buffers[n];
    if (!hclib::upcxx::is_memory_shared_with(level_c->interpolation.send_ranks[n])) {
      events[n] = hclib::upcxx::async_copy(p1, p2,
              level_c->interpolation.send_sizes[n]);
    } else {
      int rid = level_c->interpolation.send_ranks[n];
      int pos = level_c->interpolation.send_match_pos[n];
      size_t nth = MAX_NBGS* id_f; nth = 0;
      int *p = (int *) level_c->interpolation.match_rflag[n]; *(p+nth+pos) = 1;
      nshm++;
    }
  }
#endif
  _timeEnd = CycleTime();
  level_f->cycles.interpolation_send += (_timeEnd-_timeStart);

  // perform local interpolation... try and hide within Isend latency... 
  _timeStart = CycleTime();
  // PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_c->interpolation.num_blocks[1])
  hclib::future_t *fut2 = parallel_across_blocks(level_f, buffer, level_c->interpolation.num_blocks[1],
          [&level_f, &id_f, &prescale_f, &level_c, &id_c] (int buffer) {
    InterpolateBlock_PC(level_f,id_f,prescale_f,level_c,id_c,
        &level_c->interpolation.blocks[1][buffer]);
  });
  fut2->wait();
  _timeEnd = CycleTime();
  level_f->cycles.interpolation_local += (_timeEnd-_timeStart);

  // wait for MPI to finish...
  _timeStart = CycleTime();

#ifdef USE_UPCXX

  hclib::upcxx::remote_finish([&] {
      for(int n=0;n<level_c->interpolation.num_sends;n++){
        int rid = level_c->interpolation.send_ranks[n];

        if (!hclib::upcxx::is_memory_shared_with(rid)) {
          int cnt = level_c->interpolation.send_sizes[n];
          int pos = level_c->interpolation.send_match_pos[n];
          const int my_rank_copy = level_c->my_rank;
          const int depth_copy = level_f->depth;
          hclib::upcxx::async_after(rid, events[n],
                  [=] {
                      cb_unpack_int(my_rank_copy, pos, depth_copy, id_f,
                          prescale_f);
                  });
        }
      }

      free(events);
  });

  if (level_f->interpolation.num_recvs > 0) {
  size_t nth = MAX_NBGS*id_f; nth = 0;
  int *p = (int *) level_f->interpolation.rflag;
  while (1) {
    int arrived = 0;
    for (int n = 0; n < level_f->interpolation.num_recvs; n++) {
      if (level_f->interpolation.rflag[nth+n]==1) arrived++;
    }
    if (arrived == level_f->interpolation.num_recvs) break;
    hclib::upcxx::advance();
  }
  for (int n = 0; n < level_f->interpolation.num_recvs; n++) {
    p[nth+n] = 0;
  }
  }

#endif
  _timeEnd = CycleTime();
  level_f->cycles.interpolation_wait += (_timeEnd-_timeStart);

  level_f->cycles.interpolation_total += (uint64_t)(CycleTime()-_timeCommunicationStart);
}
