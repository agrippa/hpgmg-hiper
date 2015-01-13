//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <math.h>
//------------------------------------------------------------------------------------------------------------------------------
static inline void InterpolateBlock_PQ(level_type *level_f, int id_f, double prescale_f, level_type *level_c, int id_c, blockCopy_type *block){
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
    global_ptr<double> gp = lbox->vectors[id_c] + lbox->ghosts*(1+lbox->jStride+lbox->kStride); 
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
    global_ptr<double> gp = lbox->vectors[id_f] + lbox->ghosts*(1+lbox->jStride+lbox->kStride); 
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
  double OneOver32Cubed = 1.0/32/32/32;
  for(k=0;k<write_dim_k;k++){int delta_k=-read_kStride;if(k&0x1)delta_k=read_kStride;
  for(j=0;j<write_dim_j;j++){int delta_j=-read_jStride;if(j&0x1)delta_j=read_jStride;
  for(i=0;i<write_dim_i;i++){int delta_i=           -1;if(i&0x1)delta_i=           1; // i.e. even points look backwards while odd points look forward
    int write_ijk = ((i   )+write_i) + (((j   )+write_j)*write_jStride) + (((k   )+write_k)*write_kStride);
    int  read_ijk = ((i>>1)+ read_i) + (((j>>1)+ read_j)* read_jStride) + (((k>>1)+ read_k)* read_kStride);

    write[write_ijk] = prescale_f*write[write_ijk] +
                       OneOver32Cubed*(
                         -27.0*read[read_ijk-delta_i-delta_j-delta_k] +
                         270.0*read[read_ijk        -delta_j-delta_k] +
                          45.0*read[read_ijk+delta_i-delta_j-delta_k] +
                         270.0*read[read_ijk-delta_i        -delta_k] +
                       -2700.0*read[read_ijk                -delta_k] +
                        -450.0*read[read_ijk+delta_i        -delta_k] +
                          45.0*read[read_ijk-delta_i+delta_j-delta_k] +
                        -450.0*read[read_ijk        +delta_j-delta_k] +
                         -75.0*read[read_ijk+delta_i+delta_j-delta_k] +

                         270.0*read[read_ijk-delta_i-delta_j        ] +
                       -2700.0*read[read_ijk        -delta_j        ] +
                        -450.0*read[read_ijk+delta_i-delta_j        ] +
                       -2700.0*read[read_ijk-delta_i                ] +
                       27000.0*read[read_ijk                        ] +
                        4500.0*read[read_ijk+delta_i                ] +
                        -450.0*read[read_ijk-delta_i+delta_j        ] +
                        4500.0*read[read_ijk        +delta_j        ] +
                         750.0*read[read_ijk+delta_i+delta_j        ] +
                       
                          45.0*read[read_ijk-delta_i-delta_j+delta_k] +
                        -450.0*read[read_ijk        -delta_j+delta_k] +
                         -75.0*read[read_ijk+delta_i-delta_j+delta_k] +
                        -450.0*read[read_ijk-delta_i        +delta_k] +
                        4500.0*read[read_ijk                +delta_k] +
                         750.0*read[read_ijk+delta_i        +delta_k] +
                         -75.0*read[read_ijk-delta_i+delta_j+delta_k] +
                         750.0*read[read_ijk        +delta_j+delta_k] +
                         125.0*read[read_ijk+delta_i+delta_j+delta_k] 
                       );

  }}}

}


//------------------------------------------------------------------------------------------------------------------------------
// perform a (inter-level) piecewise linear interpolation
void interpolation_pq(level_type * level_f, int id_f, double prescale_f, level_type *level_c, int id_c){
    exchange_boundary(level_c,id_c,0);
  apply_BCs_quadratic(level_c,id_c,0);

  uint64_t _timeCommunicationStart = CycleTime();
  uint64_t _timeStart,_timeEnd;
  int buffer=0;
  int n;
  int my_tag = (level_f->tag<<4) | 0x7;

  // perform local interpolation... try and hide within Isend latency... 
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_c->interpolation.num_blocks[3])
  for(buffer=0;buffer<level_c->interpolation.num_blocks[3];buffer++){InterpolateBlock_PQ(level_f,id_f,prescale_f,level_c,id_c,&level_c->interpolation.blocks[3][buffer]);}
  _timeEnd = CycleTime();
  level_f->cycles.interpolation_shm += (_timeEnd-_timeStart);


  // pack MPI send buffers...
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_c->interpolation.num_blocks[0])
  for(buffer=0;buffer<level_c->interpolation.num_blocks[0];buffer++){InterpolateBlock_PQ(level_f,id_f,0.0,level_c,id_c,&level_c->interpolation.blocks[0][buffer]);} // !!! prescale==0 because you don't want to increment the MPI buffer
  _timeEnd = CycleTime();
  level_f->cycles.interpolation_pack += (_timeEnd-_timeStart);

  // loop through MPI send buffers and post Isend's...
  _timeStart = CycleTime();
#ifdef USE_UPCXX
  int nshm = 0;
  for(n=0;n<level_c->interpolation.num_sends;n++){
    global_ptr<double> p1, p2;
    p1 = level_c->interpolation.global_send_buffers[n];
    p2 = level_c->interpolation.global_match_buffers[n];

    if (!is_memory_shared_with(level_c->interpolation.send_ranks[n])) {
      event* copy_e = &level_c->interpolation.copy_e[n];
      upcxx::async_copy(p1, p2, level_c->interpolation.send_sizes[n], copy_e);
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
  PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_c->interpolation.num_blocks[1])
  for(buffer=0;buffer<level_c->interpolation.num_blocks[1];buffer++){InterpolateBlock_PQ(level_f,id_f,prescale_f,level_c,id_c,&level_c->interpolation.blocks[1][buffer]);}
  _timeEnd = CycleTime();
  level_f->cycles.interpolation_local += (_timeEnd-_timeStart);

  // wait for MPI to finish...
  _timeStart = CycleTime();

#ifdef USE_UPCXX

  for(n=0;n<level_c->interpolation.num_sends;n++){
    int rid = level_c->interpolation.send_ranks[n];

    if (!is_memory_shared_with(rid)) {
      int cnt = level_c->interpolation.send_sizes[n];
      int pos = level_c->interpolation.send_match_pos[n];
      event* copy_e = &level_c->interpolation.copy_e[n];
      event* data_e = &level_c->interpolation.data_e[n];
      async_after(rid, copy_e, data_e)(cb_unpack_int, level_c->my_rank, pos,
                  level_f->depth, id_f, prescale_f);
    }
  }

  async_wait();

  if (level_f->interpolation.num_recvs > 0) {
  size_t nth = MAX_NBGS*id_f;  nth = 0;
  int *p = (int *) level_f->interpolation.rflag;
  while (1) {
    int arrived = 0;
    for (int n = 0; n < level_f->interpolation.num_recvs; n++) {
      if (level_f->interpolation.rflag[nth+n]==1) arrived++;
    }
    if (arrived == level_f->interpolation.num_recvs) break;
    upcxx::advance();
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
