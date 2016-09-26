//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <math.h>
//------------------------------------------------------------------------------------------------------------------------------
static inline void InterpolateBlock_PL(level_type *level_f, int id_f, double prescale_f, level_type *level_c, int id_c, blockCopy_type *block){
  // interpolate 3D array from read_i,j,k of read[] to write_i,j,k in write[]
  int write_dim_i   = block->dim.i<<1; // calculate the dimensions of the resultant fine block
  int write_dim_j   = block->dim.j<<1;
  int write_dim_k   = block->dim.k<<1;

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
#else  // USE_UPCXX
     read = level_c->my_boxes[ block->read.box].vectors[        id_c] + level_c->my_boxes[ block->read.box].ghosts*(1+level_c->my_boxes[ block->read.box].jStride+level_c->my_boxes[ block->read.box].kStride);
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
  for(k=0;k<write_dim_k;k++){int delta_k=-read_kStride;if(k&0x1)delta_k=read_kStride;
  for(j=0;j<write_dim_j;j++){int delta_j=-read_jStride;if(j&0x1)delta_j=read_jStride;
  for(i=0;i<write_dim_i;i++){int delta_i=           -1;if(i&0x1)delta_i=           1; // i.e. even points look backwards while odd points look forward
    int write_ijk = ((i   )+write_i) + (((j   )+write_j)*write_jStride) + (((k   )+write_k)*write_kStride);
    int  read_ijk = ((i>>1)+ read_i) + (((j>>1)+ read_j)* read_jStride) + (((k>>1)+ read_k)* read_kStride);
    //
    // |   o   |   o   |
    // +---+---+---+---+
    // |   | x | x |   |
    //
    // CAREFUL !!!  you must guarantee you zero'd the MPI buffers(write[]) and destination boxes at some point to avoid 0.0*NaN or 0.0*inf
    // piecewise linear interpolation... NOTE, BC's must have been previously applied
    write[write_ijk] = prescale_f*write[write_ijk] + 
        0.421875*read[read_ijk                        ] +
        0.140625*read[read_ijk                +delta_k] +
        0.140625*read[read_ijk        +delta_j        ] +
        0.046875*read[read_ijk        +delta_j+delta_k] +
        0.140625*read[read_ijk+delta_i                ] +
        0.046875*read[read_ijk+delta_i        +delta_k] +
        0.046875*read[read_ijk+delta_i+delta_j        ] +
        0.015625*read[read_ijk+delta_i+delta_j+delta_k];
  }}}

}


//------------------------------------------------------------------------------------------------------------------------------
// perform a (inter-level) piecewise linear interpolation
void interpolation_pl(level_type * level_f, int id_f, double prescale_f, level_type *level_c, int id_c){
  exchange_boundary(level_c,id_c,0);
   apply_BCs_linear(level_c,id_c,0);

  uint64_t _timeCommunicationStart = CycleTime();
  uint64_t _timeStart,_timeEnd;
  int buffer=0;
  int n;
  int my_tag = (level_f->tag<<4) | 0x7;

  // perform local interpolation... try and hide within Isend latency... 
  _timeStart = CycleTime();
  // PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_c->interpolation.num_blocks[3])
  hclib::future_t *fut0 = parallel_across_blocks(level_f, buffer, level_c->interpolation.num_blocks[3],
          [&level_f, &id_f, &prescale_f, &level_c, &id_c] (int buffer) {
    InterpolateBlock_PL(level_f,id_f,prescale_f,level_c,id_c,
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
    InterpolateBlock_PL(level_f,id_f,0.0,level_c,id_c,
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
    InterpolateBlock_PL(level_f,id_f,prescale_f,level_c,id_c,
        &level_c->interpolation.blocks[1][buffer]);
  });
  fut2->wait();
  _timeEnd = CycleTime();
  level_f->cycles.interpolation_local += (_timeEnd-_timeStart);

  // wait for MPI to finish...
  _timeStart = CycleTime();

#ifdef USE_UPCXX

  for(n=0;n<level_c->interpolation.num_sends;n++){
    int rid = level_c->interpolation.send_ranks[n];

    if (!hclib::upcxx::is_memory_shared_with(rid)) {
      int cnt = level_c->interpolation.send_sizes[n];
      int pos = level_c->interpolation.send_match_pos[n];
      const int my_rank_copy = level_c->my_rank;
      const int f_depth_copy = level_f->depth;
      events[n] = hclib::upcxx::async_after(rid, events[n], [=] {
                  cb_unpack_int(my_rank_copy, pos, f_depth_copy, id_f,
                      prescale_f);
              });
    }
  }

  for (n=0;n<level_c->interpolation.num_sends;n++) {
      if (events[n]) events[n]->wait();
  }
  free(events);

  hclib::upcxx::async_wait();

  if (level_f->interpolation.num_recvs > 0) {
  size_t nth = MAX_NBGS*id_f;  nth = 0;
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
