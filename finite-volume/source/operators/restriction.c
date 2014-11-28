//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
static level_type *para_level;
extern mg_type all_grids;

void cb_copy_res(double *buf, int n, int srcid, int id_f, int depth_f, int type, int depth_c) {

  uint64_t _timeCommunicationStart = CycleTime();
  uint64_t _timeStart,_timeEnd;

  level_type *level = para_level;
  level_type *level_c;
  int buffer;

  _timeStart = CycleTime();

  //shan: define all_grids earlier in hpgmg.c
  if (all_grids.levels != NULL) {
     level = all_grids.levels[depth_f];
     level_c = all_grids.levels[depth_c];
  }
  else {
    print("WRONG! This should not happen! %d %d\n", depth_f, depth_c);
     level = para_level;
  }

  int i;
  int nth = depth * 20 + id_f;
  for (i = 0; i < level->restriction[type].num_recvs; i++) {
     if (level->restriction[type].recv_ranks[i] == srcid) {
        if (level->restriction[type].flag_data[nth][i] != 0) {
	  printf("Wrong in Ping Res Handler Proc %d recv msg from %d for id_f %d val %d\n", MYTHREAD, srcid, id_f, level->restriction[type].flag_data[nth][i]);
	}
	else {
	  level->restriction[type].flag_data[nth][i] =1;
	}
	break;
     }
  }

  int msize = gasnet_AMMaxMedium();
  int bstart = level->restriction[type].sblock2[i];
  int bend   = level->restriction[type].sblock2[i+1];

  PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_c->restriction[restrictionType].num_blocks[2])
  for(buffer=0;buffer<level_c->restriction[restrictionType].num_blocks[2];buffer++){CopyBlock(level_c,id_c,&level_c->restriction[restrictionType].blocks[2][buffer], NULL, 0);}

  if (n < msize) { // medium AM 
    PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer, bend-bstart)
    for(buffer=bstart;buffer<bend;buffer++){
      CopyBlock(level_c,id_c,&level->restriction[type].blocks[2][buffer], buf, 1);
    }
  }
  else { // long AM
    PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,bend-bstart)
    for(buffer=bstart;buffer<bend;buffer++){
      CopyBlock(level_c,id_c,&level->restriction[type].blocks[2][buffer], buf, 0);
    }
  }

  _timeEnd = CycleTime();
  level->cycles.ghostZone_unpack += (_timeEnd-_timeStart);

}

void sendNbgrDataRes(int rid, global_ptr<double> src, global_ptr<double> dest, int nelem) {

  int myid = gasnet_mynode(); 
  double * lsrc = (double *)src.raw_ptr();
  double * ldst = (double *)dest.raw_ptr();
  int msize = gasnet_AMMaxMedium();

  if (nelem * sizeof(double) < msize) {
     // using mediumAM
    GASNET_Safe(gasnet_AMRequestMedium4(rid, P2P_PING_RES_MEDREQUEST, lsrc, nelem*sizeof(double), para_id_f, para_level_f->depth,  para_restrict_type, para_level_c->depth));
  }
  else {
    // using longAM
    GASNET_Safe(gasnet_AMRequestLongAsync4(rid, P2P_PING_RES_LONGREQUEST, lsrc, nelem*sizeof(double), ldst, para_id_f, para_level_f->depth, para_restrict_type, para_level_c->depth));
  }

}

void syncNeighborRes(int nbgr, int vid) {
  GASNET_BLOCKUNTIL(upcxx::p2p_flag_res[vid] == nbgr);
  upcxx::p2p_flag_res[vid] = 0;
}

static inline void RestrictBlock(level_type *level_c, int id_c, level_type *level_f, int id_f, blockCopy_type *block, int restrictionType){
  // restrict 3D array from read_i,j,k of read[] to write_i,j,k in write[]
  int   dim_i       = block->dim.i; // calculate the dimensions of the resultant coarse block
  int   dim_j       = block->dim.j;
  int   dim_k       = block->dim.k;

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
     read = level_f->my_boxes[ block->read.box].vectors[id_f] + level_f->my_boxes[ block->read.box].ghosts*(1+level_f->my_boxes[ block->read.box].jStride+level_f->my_boxes[ block->read.box].kStride);
     read_jStride = level_f->my_boxes[block->read.box ].jStride;
     read_kStride = level_f->my_boxes[block->read.box ].kStride;
  }
  if(block->write.box>=0){
    write = level_c->my_boxes[block->write.box].vectors[id_c] + level_c->my_boxes[block->write.box].ghosts*(1+level_c->my_boxes[block->write.box].jStride+level_c->my_boxes[block->write.box].kStride);
    write_jStride = level_c->my_boxes[block->write.box].jStride;
    write_kStride = level_c->my_boxes[block->write.box].kStride;
  }



  int i,j,k;
  switch(restrictionType){
    case RESTRICT_CELL:
         for(k=0;k<dim_k;k++){
         for(j=0;j<dim_j;j++){
         for(i=0;i<dim_i;i++){
           int write_ijk = ((i   )+write_i) + ((j   )+write_j)*write_jStride + ((k   )+write_k)*write_kStride;
           int  read_ijk = ((i<<1)+ read_i) + ((j<<1)+ read_j)* read_jStride + ((k<<1)+ read_k)* read_kStride;
           write[write_ijk] = ( read[read_ijk                            ]+read[read_ijk+1                          ] +
                                read[read_ijk  +read_jStride             ]+read[read_ijk+1+read_jStride             ] +
                                read[read_ijk               +read_kStride]+read[read_ijk+1             +read_kStride] +
                                read[read_ijk  +read_jStride+read_kStride]+read[read_ijk+1+read_jStride+read_kStride] ) * 0.125;
         }}}break;
    case RESTRICT_FACE_I:
         for(k=0;k<dim_k;k++){
         for(j=0;j<dim_j;j++){
         for(i=0;i<dim_i;i++){
           int write_ijk = ((i   )+write_i) + ((j   )+write_j)*write_jStride + ((k   )+write_k)*write_kStride;
           int  read_ijk = ((i<<1)+ read_i) + ((j<<1)+ read_j)* read_jStride + ((k<<1)+ read_k)* read_kStride;
           write[write_ijk] = ( read[read_ijk                          ] +
                                read[read_ijk+read_jStride             ] +
                                read[read_ijk             +read_kStride] +
                                read[read_ijk+read_jStride+read_kStride] ) * 0.25;
         }}}break;
    case RESTRICT_FACE_J:
         for(k=0;k<dim_k;k++){
         for(j=0;j<dim_j;j++){
         for(i=0;i<dim_i;i++){
           int write_ijk = ((i   )+write_i) + ((j   )+write_j)*write_jStride + ((k   )+write_k)*write_kStride;
           int  read_ijk = ((i<<1)+ read_i) + ((j<<1)+ read_j)* read_jStride + ((k<<1)+ read_k)* read_kStride;
           write[write_ijk] = ( read[read_ijk               ] +
                                read[read_ijk+1             ] +
                                read[read_ijk  +read_kStride] +
                                read[read_ijk+1+read_kStride] ) * 0.25;
         }}}break;
    case RESTRICT_FACE_K:
         for(k=0;k<dim_k;k++){
         for(j=0;j<dim_j;j++){
         for(i=0;i<dim_i;i++){
           int write_ijk = ((i   )+write_i) + ((j   )+write_j)*write_jStride + ((k   )+write_k)*write_kStride;
           int  read_ijk = ((i<<1)+ read_i) + ((j<<1)+ read_j)* read_jStride + ((k<<1)+ read_k)* read_kStride;
           write[write_ijk] = ( read[read_ijk               ] +
                                read[read_ijk+1             ] +
                                read[read_ijk  +read_jStride] +
                                read[read_ijk+1+read_jStride] ) * 0.25;
         }}}break;
  }

}


//------------------------------------------------------------------------------------------------------------------------------
// perform a (inter-level) restriction
void restriction(level_type * level_c, int id_c, level_type *level_f, int id_f, int restrictionType){
  uint64_t _timeCommunicationStart = CycleTime();
  uint64_t _timeStart,_timeEnd;
  int buffer=0;
  int n;
  int my_tag = (level_f->tag<<4) | 0x5;

#ifdef UPCXX_P2P
  para_id_c = id_c;
  para_id_f = id_f;
  para_level_c = level_c;
  para_level_f = level_f;
  para_restrict_type = restrictionType;
#endif

#ifdef USE_UPCXX
  _timeStart = CycleTime();
#ifndef UPCXX_P2P
#ifdef USE_SUBCOMM
  // do we need barrier from both level ?
  MPI_Barrier(level_f->MPI_COMM_ALLREDUCE);
#else
  upcxx::barrier();
#endif
#endif
  _timeEnd = CycleTime();
  level_f->cycles.restriction_recv += (_timeEnd-_timeStart);
#elif USE_MPI

  // by convention, level_f allocates a combined array of requests for both level_f sends and level_c recvs...
  int nMessages = level_c->restriction[restrictionType].num_recvs + level_f->restriction[restrictionType].num_sends;
  MPI_Request *recv_requests = level_f->restriction[restrictionType].requests;
  MPI_Request *send_requests = level_f->restriction[restrictionType].requests + level_c->restriction[restrictionType].num_recvs;


  // loop through packed list of MPI receives and prepost Irecv's...
  _timeStart = CycleTime();
  #ifdef USE_MPI_THREAD_MULTIPLE
  #pragma omp parallel for schedule(dynamic,1)
  #endif
  for(n=0;n<level_c->restriction[restrictionType].num_recvs;n++){
    MPI_Irecv(level_c->restriction[restrictionType].recv_buffers[n],
              level_c->restriction[restrictionType].recv_sizes[n],
              MPI_DOUBLE,
              level_c->restriction[restrictionType].recv_ranks[n],
              my_tag,
              MPI_COMM_WORLD,
              &recv_requests[n]
    );
  }
  _timeEnd = CycleTime();
  level_f->cycles.restriction_recv += (_timeEnd-_timeStart);
#endif

  // pack MPI send buffers...
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_f->restriction[restrictionType].num_blocks[0])
  for(buffer=0;buffer<level_f->restriction[restrictionType].num_blocks[0];buffer++){RestrictBlock(level_c,id_c,level_f,id_f,&level_f->restriction[restrictionType].blocks[0][buffer],restrictionType);}
  _timeEnd = CycleTime();
  level_f->cycles.restriction_pack += (_timeEnd-_timeStart);

 
  // loop through MPI send buffers and post Isend's...
  _timeStart = CycleTime();
#ifdef USE_UPCXX
  for(n=0;n<level_f->restriction[restrictionType].num_sends;n++){
    global_ptr<double> p1, p2;
    p1 = level_f->restriction[restrictionType].global_send_buffers[n];
    p2 = level_f->restriction[restrictionType].global_match_buffers[n];
#ifndef UPCXX_P2P
    upcxx::async_copy(p1, p2, level_f->restriction[restrictionType].send_sizes[n]);    
#else
    sendNbgrDataRes(level->restriction[restrictionType].send_ranks[n], 
		 p1, p2, level->restriction[restrictionType].send_sizes[n]);
#endif

  }
#elif USE_MPI
  #ifdef USE_MPI_THREAD_MULTIPLE
  #pragma omp parallel for schedule(dynamic,1)
  #endif
  for(n=0;n<level_f->restriction[restrictionType].num_sends;n++){
    MPI_Isend(level_f->restriction[restrictionType].send_buffers[n],
              level_f->restriction[restrictionType].send_sizes[n],
              MPI_DOUBLE,
              level_f->restriction[restrictionType].send_ranks[n],
              my_tag,
              MPI_COMM_WORLD,
              &send_requests[n]
    );
  }
#endif
  _timeEnd = CycleTime();
  level_f->cycles.restriction_send += (_timeEnd-_timeStart);


  // perform local restriction[restrictionType]... try and hide within Isend latency... 
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_f->restriction[restrictionType].num_blocks[1])
  for(buffer=0;buffer<level_f->restriction[restrictionType].num_blocks[1];buffer++){RestrictBlock(level_c,id_c,level_f,id_f,&level_f->restriction[restrictionType].blocks[1][buffer],restrictionType);}
  _timeEnd = CycleTime();
  level_f->cycles.restriction_local += (_timeEnd-_timeStart);


  // wait for MPI to finish...
  _timeStart = CycleTime();

  int nth = level->depth * 20 + id_f;
  while (1) {
    int arrived = 0;
    for (int n = 0; n < level->restriction[restrictionType].num_recvs; n++) {
      if (level->restriction[restrictionType].flag_data[nth][n]==1) arrived++;
    }
    if (arrived == level->restriction[restrictionType].num_recvs) break;
    upcxx::advance();
  }
  for (int n = 0; n < level->restriction[restrictionType].num_recvs; n++) {
    level->restriction[restrictionType].flag_data[nth][n] = 0;
  }

#ifdef USE_UPCXX
#ifndef UPCXX_P2P
  async_copy_fence();
#ifdef USE_SUBCOMM
  MPI_Barrier(level_f->MPI_COMM_ALLREDUCE);
#else
  upcxx::barrier();
#endif
#else
  syncNeighborRes(level->restriction[restrictionType].num_sends, id_f);
#endif

#elif USE_MPI
  if(nMessages)MPI_Waitall(nMessages,level_f->restriction[restrictionType].requests,level_f->restriction[restrictionType].status);
#endif

  _timeEnd = CycleTime();
  level_f->cycles.restriction_wait += (_timeEnd-_timeStart);

#ifndef UPCXX_P2P
  // unpack MPI receive buffers 
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_c->restriction[restrictionType].num_blocks[2])
  for(buffer=0;buffer<level_c->restriction[restrictionType].num_blocks[2];buffer++){CopyBlock(level_c,id_c,&level_c->restriction[restrictionType].blocks[2][buffer], NULL, 0);}
  _timeEnd = CycleTime();
  level_f->cycles.restriction_unpack += (_timeEnd-_timeStart);
 #endif

  level_f->cycles.restriction_total += (uint64_t)(CycleTime()-_timeCommunicationStart);
}
