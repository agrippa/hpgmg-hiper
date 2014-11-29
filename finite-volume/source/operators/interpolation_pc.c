//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
static double para_prescale_f[20]={1.0};

extern mg_type all_grids;

void cb_copy_int(double *buf, int n, int srcid, int depth_f, int id_f, int myid, int id_c, int depth_c) {

  uint64_t _timeCommunicationStart = CycleTime();
  uint64_t _timeStart,_timeEnd;

  level_type *level_f;
  level_type *level_c;
  double prescale_f;
  int buffer;

  _timeStart = CycleTime();

  //shan: define all_grids earlier in hpgmg.c
  if (all_grids.levels != NULL) {
     level_f = all_grids.levels[depth_f];
     level_c = all_grids.levels[depth_c];
  }
  else {
    printf("WRONG! This should not happen! %d %d\n", depth_f, depth_c);
  }
  prescale_f = para_prescale_f[level_f->depth];

  int i;
  int nth = depth_f * 20 + id_f;
  for (i = 0; i < level_f->interpolation.num_recvs; i++) {
     if (level_f->interpolation.recv_ranks[i] == srcid) {
        if (level_f->interpolation.flag_data[nth][i] != 0) {
          printf("Wrong in Ping Res Handler Proc %d recv msg from %d for id_f %d val %d\n", MYTHREAD, srcid, id_f, level_c->interpolation.flag_data[nth][i]);
        }
        else {
          level_f->interpolation.flag_data[nth][i] =1;
        }
        break;
     }
  }

/**
  printf("NN i is %d limit is %d in level_f %d id_f %d level_c %d id_c %d for proc %d from %d %d\n",
                  i, level_f->interpolation.num_recvs, level_f->depth, id_f, level_c->depth, id_c, MYTHREAD, srcid, myid);
**/
  int msize = gasnet_AMMaxMedium();
  int bstart = level_f->interpolation.sblock2[i];
  int bend   = level_f->interpolation.sblock2[i+1];

  if (n < msize) { // medium AM 
//    PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer, bend-bstart)
    for(buffer=bstart;buffer<bend;buffer++){
      IncrementBlock(level_f,id_f,prescale_f,&level_f->interpolation.blocks[2][buffer], buf, 1);
    }
  }
  else { // long AM
//    PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,bend-bstart)
    for(buffer=bstart;buffer<bend;buffer++){
      IncrementBlock(level_f,id_f,prescale_f,&level_f->interpolation.blocks[2][buffer], buf, 0);
    }
  }

  _timeEnd = CycleTime();
  level_f->cycles.interpolation_unpack += (_timeEnd-_timeStart);

}

void sendNbgrDataInt(int rid, global_ptr<double> src, global_ptr<double> dest, int nelem, int depth_f, int id_f, int id_c, int depth_c) {

  int myid = gasnet_mynode();
  double * lsrc = (double *)src.raw_ptr();
  double * ldst = (double *)dest.raw_ptr();
  int msize = gasnet_AMMaxMedium();

  if (nelem * sizeof(double) < msize) {
     // using mediumAM
    GASNET_Safe(gasnet_AMRequestMedium5(rid, P2P_INT_MEDREQUEST, lsrc, nelem*sizeof(double), depth_f, id_f,  myid , id_c, depth_c));
  }
  else {
    // using longAM
    GASNET_Safe(gasnet_AMRequestLongAsync5(rid, P2P_INT_LONGREQUEST, lsrc, nelem*sizeof(double), ldst, depth_f, id_f, myid, id_c, depth_c));
  }
/**
  printf("SEND proc %d to %d level_f %d id_f %d level_c %d id_c %d\n", MYTHREAD, rid, depth_f, id_f, depth_c, id_c);
**/

}

void syncNeighborInt(int nbgr, int vid) {
  GASNET_BLOCKUNTIL(upcxx::p2p_flag_int[vid] == nbgr);
  upcxx::p2p_flag_int[vid] = 0;
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
     read = level_c->my_boxes[ block->read.box].vectors[id_c] + level_c->my_boxes[ block->read.box].ghosts*(1+level_c->my_boxes[ block->read.box].jStride+level_c->my_boxes[ block->read.box].kStride);
     read_jStride = level_c->my_boxes[block->read.box ].jStride;
     read_kStride = level_c->my_boxes[block->read.box ].kStride;
  }
  if(block->write.box>=0){
    write = level_f->my_boxes[block->write.box].vectors[id_f] + level_f->my_boxes[block->write.box].ghosts*(1+level_f->my_boxes[block->write.box].jStride+level_f->my_boxes[block->write.box].kStride);
    write_jStride = level_f->my_boxes[block->write.box].jStride;
    write_kStride = level_f->my_boxes[block->write.box].kStride;
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

#ifdef UPCXX_P2P
  para_prescale_f[level_f->depth] = prescale_f;
#endif

/**
  printf("INT proc %d level_f %d id_f %d level_c %d id_c %d nsends %d nrecv %d\n",
              MYTHREAD, level_f->depth, id_f, level_c->depth, id_c, level_c->interpolation.num_sends, level_f->interpolation.num_recvs);
**/
  _timeStart = CycleTime();
#ifdef USE_UPCXX
#ifndef UPCXX_P2P
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
  for(buffer=0;buffer<level_c->interpolation.num_blocks[0];buffer++){InterpolateBlock_PC(level_f,id_f,0.0,level_c,id_c,&level_c->interpolation.blocks[0][buffer]);} // !!! prescale==0 because you don't want to increment the MPI buffer
  _timeEnd = CycleTime();
  level_f->cycles.interpolation_pack += (_timeEnd-_timeStart);

 
  // loop through MPI send buffers and post Isend's...
  _timeStart = CycleTime();
#ifdef USE_UPCXX
  for(n=0;n<level_c->interpolation.num_sends;n++){
    global_ptr<double> p1, p2;
    p1 = level_c->interpolation.global_send_buffers[n];
    p2 = level_c->interpolation.global_match_buffers[n];
#ifndef UPCXX_P2P
    upcxx::async_copy(p1, p2, level_c->interpolation.send_sizes[n]);
#else
    sendNbgrDataInt(level_c->interpolation.send_ranks[n], p1, p2, level_c->interpolation.send_sizes[n], level_f->depth, id_f, id_c, level_c->depth);
#endif
  }
#endif
  _timeEnd = CycleTime();
  level_f->cycles.interpolation_send += (_timeEnd-_timeStart);


  // perform local interpolation... try and hide within Isend latency... 
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_c->interpolation.num_blocks[1])
  for(buffer=0;buffer<level_c->interpolation.num_blocks[1];buffer++){InterpolateBlock_PC(level_f,id_f,prescale_f,level_c,id_c,&level_c->interpolation.blocks[1][buffer]);}
  _timeEnd = CycleTime();
  level_f->cycles.interpolation_local += (_timeEnd-_timeStart);


  // wait for MPI to finish...
  _timeStart = CycleTime();

#ifdef USE_UPCXX
#ifdef UPCXX_P2P

  int nth = level_f->depth * 20 + id_f;

/**
  if (MYTHREAD == 227 || MYTHREAD == 208 || MYTHREAD == 161)
  printf("ENTER WAIT proc %d level_f %d id_f %d nrecv %d\n first %d", MYTHREAD, level_f->depth, id_f, level_f->interpolation.num_recvs, level_f->interpolation.flag_data[nth][0]);
**/
  while (1) {
    int arrived = 0;
    for (int n = 0; n < level_f->interpolation.num_recvs; n++) {
      if (level_f->interpolation.flag_data[nth][n]==1) arrived++;
    }
    if (arrived == level_f->interpolation.num_recvs) break;
    upcxx::advance();
    gasnet_AMPoll();
  }
  for (int n = 0; n < level_f->interpolation.num_recvs; n++) {
    level_f->interpolation.flag_data[nth][n] = 0;
  }
/**
  if (MYTHREAD == 227 || MYTHREAD == 208 || MYTHREAD == 161)
  printf("PASS WAIT proc %d level_f %d id_f %d nrecv %d\n first %d", MYTHREAD, level_f->depth, id_f, level_f->interpolation.num_recvs, level_f->interpolation.flag_data[nth][0]);
**/

  syncNeighborInt(level_c->interpolation.num_sends, id_c);

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
#ifndef UPCXX_P2P 
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_f->interpolation.num_blocks[2])
  for(buffer=0;buffer<level_f->interpolation.num_blocks[2];buffer++){IncrementBlock(level_f,id_f,prescale_f,&level_f->interpolation.blocks[2][buffer],NULL,0);}
  _timeEnd = CycleTime();
  level_f->cycles.interpolation_unpack += (_timeEnd-_timeStart);
#endif

  level_f->cycles.interpolation_total += (uint64_t)(CycleTime()-_timeCommunicationStart);
}
