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
#ifdef UPCXX_SHARED
    int rank = level_c->rank_of_box[block->read.box];
    if (!upcxx::is_memory_shared_with(rank)) {
      printf("WrongPL: Proc %d level %d read box %d rank is %d not shared!\n",level_c->my_rank,level_c->depth,block->read.box,rank);
      exit(1);
    }
    global_ptr<box_type> box = level_c->addr_of_box[block->read.box];
    box_type *lbox = (box_type *) box;
    global_ptr<double> gp = lbox->vectors[id_c] + lbox->ghosts*(1+lbox->jStride+lbox->kStride); 
    read = (double *)gp;
    read_jStride = lbox->jStride;
    read_kStride = lbox->kStride;
#else
     box_type *lbox = &(level_c->my_boxes[block->read.box]);
     read = lbox->vectors[id_c] + lbox->ghosts*(1+lbox->jStride+lbox->kStride);
     read_jStride = lbox->jStride;
     read_kStride = lbox->kStride;
#endif // UPCXX_SHARED

#else  // USE_UPCXX
     read = level_c->my_boxes[ block->read.box].vectors[        id_c] + level_c->my_boxes[ block->read.box].ghosts*(1+level_c->my_boxes[ block->read.box].jStride+level_c->my_boxes[ block->read.box].kStride);
     read_jStride = level_c->my_boxes[block->read.box ].jStride;
     read_kStride = level_c->my_boxes[block->read.box ].kStride;
#endif
  }
  if(block->write.box>=0){
#ifdef USE_UPCXX
#ifdef UPCXX_SHARED
    int rank = level_f->rank_of_box[block->write.box];
    if (!upcxx::is_memory_shared_with(rank)) {
      printf("WrongPL: Proc %d level %d write box %d rank is %d not shared!\n",level_f->my_rank,level_f->depth,block->write.box,rank);
      exit(1);
    }
    global_ptr<box_type> box = level_f->addr_of_box[block->write.box];
    box_type *lbox = (box_type *) box;
    global_ptr<double> gp = lbox->vectors[id_f] + lbox->ghosts*(1+lbox->jStride+lbox->kStride); 
    write = (double *)gp;
    write_jStride = lbox->jStride;
    write_kStride = lbox->kStride;
#else
    box_type *lbox = &(level_f->my_boxes[block->write.box]);    
    write = lbox->vectors[id_f] + lbox->ghosts*(1+lbox->jStride+lbox->kStride);
    write_jStride = lbox->jStride;
    write_kStride = lbox->kStride;
#endif // UPCXX_SHARED

#else  // USE_UPCXX
    write = level_f->my_boxes[block->write.box].vectors[id_f] + level_f->my_boxes[block->write.box].ghosts*(1+level_f->my_boxes[block->write.box].jStride+level_f->my_boxes[block->write.box].kStride);
    write_jStride = level_f->my_boxes[block->write.box].jStride;
    write_kStride = level_f->my_boxes[block->write.box].kStride;
#endif
  }
 
 
  int i,j,k;
  for(k=0;k<write_dim_k;k++){
  for(j=0;j<write_dim_j;j++){
  for(i=0;i<write_dim_i;i++){
    int write_ijk = ((i   )+write_i) + (((j   )+write_j)*write_jStride) + (((k   )+write_k)*write_kStride);
    int  read_ijk = ((i>>1)+ read_i) + (((j>>1)+ read_j)* read_jStride) + (((k>>1)+ read_k)* read_kStride);
    //
    // |   o   |   o   |
    // +---+---+---+---+
    // |   | x | x |   |
    //
    // CAREFUL !!!  you must guarantee you zero'd the MPI buffers(write[]) and destination boxes at some point to avoid 0.0*NaN or 0.0*inf
    // piecewise linear interpolation... NOTE, BC's must have been previously applied
    int delta_i=           -1;if(i&0x1)delta_i=           1; // i.e. even points look backwards while odd points look forward
    int delta_j=-read_jStride;if(j&0x1)delta_j=read_jStride;
    int delta_k=-read_kStride;if(k&0x1)delta_k=read_kStride;
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
   apply_BCs_linear(level_c,id_c);

  uint64_t _timeCommunicationStart = CycleTime();
  uint64_t _timeStart,_timeEnd;
  int buffer=0;
  int n;
  int my_tag = (level_f->tag<<4) | 0x7;

#ifdef UPCXX_AM
  level_f->prescale_fl = prescale_f;
#endif

  _timeStart = CycleTime();
#ifdef USE_UPCXX
#ifndef UPCXX_AM
#ifdef USE_SUBCOMM
  MPI_Barrier(level_f->MPI_COMM_ALLREDUCE);
#else
  upcxx::barrier();
#endif
#endif
#endif
  _timeEnd = CycleTime();
  level_f->cycles.interpolation_recv += (_timeEnd-_timeStart);

  // pack MPI send buffers...
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_c->interpolation.num_blocks[0])
  for(buffer=0;buffer<level_c->interpolation.num_blocks[0];buffer++){InterpolateBlock_PL(level_f,id_f,0.0,level_c,id_c,&level_c->interpolation.blocks[0][buffer]);} // !!! prescale==0 because you don't want to increment the MPI buffer
  _timeEnd = CycleTime();
  level_f->cycles.interpolation_pack += (_timeEnd-_timeStart);

 
  // loop through MPI send buffers and post Isend's...
  _timeStart = CycleTime();
#ifdef USE_UPCXX
  for(n=0;n<level_c->interpolation.num_sends;n++){
    global_ptr<double> p1, p2;
    p1 = level_c->interpolation.global_send_buffers[n];
    p2 = level_c->interpolation.global_match_buffers[n];
#ifndef UPCXX_AM
    upcxx::async_copy(p1, p2, level_c->interpolation.send_sizes[n]);
#else
    sendNbgrDataInt(level_c->interpolation.send_ranks[n], p1, p2, level_c->interpolation.send_sizes[n], level_f->depth, id_f, id_c, level_c->depth,1);
#endif
  }
#endif
  _timeEnd = CycleTime();
  level_f->cycles.interpolation_send += (_timeEnd-_timeStart);


  // perform local interpolation... try and hide within Isend latency... 
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_c->interpolation.num_blocks[1])
  for(buffer=0;buffer<level_c->interpolation.num_blocks[1];buffer++){InterpolateBlock_PL(level_f,id_f,prescale_f,level_c,id_c,&level_c->interpolation.blocks[1][buffer]);}
  _timeEnd = CycleTime();
  level_f->cycles.interpolation_local += (_timeEnd-_timeStart);

  // wait for MPI to finish...
  _timeStart = CycleTime();

#ifdef USE_UPCXX
#ifdef UPCXX_AM

printf("MPL proc %d enter wait in level %d id %d num %d\n", level_f->my_rank, level_f->depth, id_f, level_f->interpolation.num_recvs);

  while (1) {
    int arrived = 0;
    for (int n = 0; n < level_f->interpolation.num_recvs; n++) {
      if (level_f->interpolation.rflag[id_f*2+1][n]==1) arrived++;
    }
    if (arrived == level_f->interpolation.num_recvs) break;
    upcxx::advance();
    gasnet_AMPoll();
  }
  for (int n = 0; n < level_f->interpolation.num_recvs; n++) {
    level_f->interpolation.rflag[id_f*2+1][n] = 0;
  }

printf("MPL proc %d pass  wait in level %d id %d num %d\n", level_f->my_rank, level_f->depth, id_f, level_f->interpolation.num_recvs);
  syncNeighborInt(level_c->interpolation.num_sends, level_c->depth, id_c, 1);

#else

  async_copy_fence();
#ifdef USE_SUBCOMM
  MPI_Barrier(level_f->MPI_COMM_ALLREDUCE);
#else
  upcxx::barrier();
#endif
#endif
#endif
  _timeEnd = CycleTime();
  level_f->cycles.interpolation_wait += (_timeEnd-_timeStart);


  // unpack MPI receive buffers 
#ifndef UPCXX_AM
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_f->interpolation.num_blocks[2])
  for(buffer=0;buffer<level_f->interpolation.num_blocks[2];buffer++){IncrementBlock(level_f,id_f,prescale_f,&level_f->interpolation.blocks[2][buffer], NULL, 0);}
  _timeEnd = CycleTime();
  level_f->cycles.interpolation_unpack += (_timeEnd-_timeStart);
#endif
 
  level_f->cycles.interpolation_total += (uint64_t)(CycleTime()-_timeCommunicationStart);
}
