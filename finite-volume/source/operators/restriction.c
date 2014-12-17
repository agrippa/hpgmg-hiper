//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------

extern mg_type all_grids;

void cb_copy_res(double *buf, int n, int srcid, int depth_f, int id_f, int type, int id_c, int depth_c) {

  uint64_t _timeCommunicationStart = CycleTime();
  uint64_t _timeStart,_timeEnd;

  level_type *level_f;
  level_type *level_c;
  int buffer;

  _timeStart = CycleTime();

  level_f = all_grids.levels[depth_f];
  level_c = all_grids.levels[depth_c];

  int i;
  for (i = 0; i < level_c->restriction[type].num_recvs; i++) {
     if (level_c->restriction[type].recv_ranks[i] == srcid) {
        if (level_c->restriction[type].rflag[id_c][i] != 0) {
	  printf("Wrong in Ping Res Handler Proc %d recv msg from %d for id_f %d val %d\n", MYTHREAD, srcid, id_f, level_c->restriction[type].rflag[id_c][i]);
	}
	else {
	  level_c->restriction[type].rflag[id_c][i] =1;
	}
	break;
     }
  }

  if (n > 0) {
  int msize = gasnet_AMMaxMedium();
  int bstart = level_c->restriction[type].sblock2[i];
  int bend   = level_c->restriction[type].sblock2[i+1];

  if (n < msize) { // medium AM 
    PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer, bend-bstart)
    for(buffer=bstart;buffer<bend;buffer++){
      CopyBlock(level_c,id_c,&level_c->restriction[type].blocks[2][buffer], buf, 11);
    }
  }
  else { // long AM
    PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,bend-bstart)
    for(buffer=bstart;buffer<bend;buffer++){
      CopyBlock(level_c,id_c,&level_c->restriction[type].blocks[2][buffer], buf, 10);
    }
  }
  }

  _timeEnd = CycleTime();
  level_f->cycles.restriction_unpack += (_timeEnd-_timeStart);

}

void sendNbgrDataRes(int rid,global_ptr<double> src,global_ptr<double> dest,int nelem,int depth_f,int id_f,int rtype,int id_c, int depth_c) {

  int myid = gasnet_mynode(); 
  double * lsrc = (double *)src.raw_ptr();
  double * ldst = (double *)dest.raw_ptr();
  int msize = gasnet_AMMaxMedium();

  if (nelem * sizeof(double) < msize) {
     // using mediumAM
    GASNET_Safe(gasnet_AMRequestMedium5(rid, P2P_RES_MEDREQUEST, lsrc, nelem*sizeof(double), depth_f, id_f, rtype, id_c, depth_c));
  }
  else {
    // using longAM
    GASNET_Safe(gasnet_AMRequestLongAsync5(rid,P2P_RES_LONGREQUEST,lsrc,nelem*sizeof(double),ldst,depth_f,id_f,rtype,id_c,depth_c));
  }

}

void syncNeighborRes(int nbgr, int depth, int vid, int rtype) {
  GASNET_BLOCKUNTIL(upcxx::p2p_flag_res[depth][vid*4+rtype] == nbgr);
  upcxx::p2p_flag_res[depth][vid*4+rtype] = 0;
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
#ifdef USE_UPCXX
#ifdef UPCXX_SHARED
     int rank = level_f->rank_of_box[block->read.box];
     if (!upcxx::is_memory_shared_with(rank)) {
      printf("WrongR: Proc %d level %d read box %d rank is %d not shared!\n",level_f->my_rank,level_f->depth,block->read.box,rank);
      exit(1);
     }
     global_ptr<box_type> box = level_f->addr_of_box[block->read.box];
     box_type *lbox = (box_type *) box;
     global_ptr<double> gp = lbox->vectors[id_f] + lbox->ghosts*(1+lbox->jStride+lbox->kStride); 
     read = (double *)gp;
     read_jStride = lbox->jStride;
     read_kStride = lbox->kStride;
#else
     box_type *lbox = &(level_f->my_boxes[block->read.box]);     
     read = lbox->vectors[id_f] + lbox->ghosts*(1+lbox->jStride+lbox->kStride);
     read_jStride = lbox->jStride;
     read_kStride = lbox->kStride;
#endif // UPCXX_SHARED

#else  // USE_UPCXX
     read = level_f->my_boxes[ block->read.box].vectors[id_f] + level_f->my_boxes[ block->read.box].ghosts*(1+level_f->my_boxes[ block->read.box].jStride+level_f->my_boxes[ block->read.box].kStride);
     read_jStride = level_f->my_boxes[block->read.box ].jStride;
     read_kStride = level_f->my_boxes[block->read.box ].kStride;
#endif
  }
  if(block->write.box>=0){
#ifdef USE_UPCXX
#ifdef UPCXX_SHARED
    int rank = level_c->rank_of_box[block->write.box];
    if (!upcxx::is_memory_shared_with(rank)) {
     printf("WrongR: Proc %d level %d write box %d rank is %d not shared!\n",level_c->my_rank,level_c->depth,block->write.box,rank);
     exit(1);
    }
    global_ptr<box_type> box = level_c->addr_of_box[block->write.box];
    box_type *lbox = (box_type *) box;
    global_ptr<double> gp = lbox->vectors[id_c] + lbox->ghosts*(1+lbox->jStride+lbox->kStride); 
    write = (double *)gp;
    write_jStride = lbox->jStride;
    write_kStride = lbox->kStride;
#else
    box_type *lbox = &(level_c->my_boxes[block->write.box]); 
    write = lbox->vectors[id_c] + lbox->ghosts*(1+lbox->jStride+lbox->kStride);
    write_jStride = lbox->jStride;
    write_kStride = lbox->kStride;
#endif  // UPCXX_SHARED

#else   // USE_UPCXX
    write = level_c->my_boxes[block->write.box].vectors[id_c] + level_c->my_boxes[block->write.box].ghosts*(1+level_c->my_boxes[block->write.box].jStride+level_c->my_boxes[block->write.box].kStride);
    write_jStride = level_c->my_boxes[block->write.box].jStride;
    write_kStride = level_c->my_boxes[block->write.box].kStride;
#endif
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

  _timeStart = CycleTime();
#ifdef USE_UPCXX
#ifndef UPCXX_AM
#ifdef USE_SUBCOMM
  // do we need barrier from both level ?
  MPI_Barrier(level_f->MPI_COMM_ALLREDUCE);
#else
  upcxx::barrier();
#endif
#endif
#endif
  _timeEnd = CycleTime();
  level_f->cycles.restriction_recv += (_timeEnd-_timeStart);

  // perform local restriction[restrictionType]... try and hide within Isend latency... 
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_f->restriction[restrictionType].num_blocks[1])
  for(buffer=0;buffer<level_f->restriction[restrictionType].num_blocks[1];buffer++){RestrictBlock(level_c,id_c,level_f,id_f,&level_f->restriction[restrictionType].blocks[1][buffer],restrictionType);}
  _timeEnd = CycleTime();
  level_f->cycles.restriction_local += (_timeEnd-_timeStart);

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
#ifndef UPCXX_AM
    upcxx::async_copy(p1, p2, level_f->restriction[restrictionType].send_sizes[n]);    
#else
    sendNbgrDataRes(level_f->restriction[restrictionType].send_ranks[n], 
		    p1, p2, level_f->restriction[restrictionType].send_sizes[n],
		    level_f->depth,id_f,restrictionType,id_c,level_c->depth);
#endif

  }
#endif
  _timeEnd = CycleTime();
  level_f->cycles.restriction_send += (_timeEnd-_timeStart);

/*****
  // perform local restriction[restrictionType]... try and hide within Isend latency... 
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_f->restriction[restrictionType].num_blocks[1])
  for(buffer=0;buffer<level_f->restriction[restrictionType].num_blocks[1];buffer++){RestrictBlock(level_c,id_c,level_f,id_f,&level_f->restriction[restrictionType].blocks[1][buffer],restrictionType);}
  _timeEnd = CycleTime();
  level_f->cycles.restriction_local += (_timeEnd-_timeStart);
*****/

  // wait for MPI to finish...
  _timeStart = CycleTime();
#ifdef UPCXX_AM
  while (1) {
    int arrived = 0;
    for (int n = 0; n < level_c->restriction[restrictionType].num_recvs; n++) {
      if (level_c->restriction[restrictionType].rflag[id_c][n]==1) arrived++;
    }
    if (arrived == level_c->restriction[restrictionType].num_recvs) break;
    upcxx::advance();
  }
  for (int n = 0; n < level_c->restriction[restrictionType].num_recvs; n++) {
    level_c->restriction[restrictionType].rflag[id_c][n] = 0;
  }

#endif

#ifdef USE_UPCXX
#ifndef UPCXX_AM
  async_copy_fence();
#ifdef USE_SUBCOMM
  MPI_Barrier(level_f->MPI_COMM_ALLREDUCE);
#else
  upcxx::barrier();
#endif
#else
  syncNeighborRes(level_f->restriction[restrictionType].num_sends, level_f->depth, id_f, restrictionType);
#endif
#endif

  _timeEnd = CycleTime();
  level_f->cycles.restriction_wait += (_timeEnd-_timeStart);

#ifndef UPCXX_AM
  // unpack MPI receive buffers 
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_c->restriction[restrictionType].num_blocks[2])
  for(buffer=0;buffer<level_c->restriction[restrictionType].num_blocks[2];buffer++){CopyBlock(level_c,id_c,&level_c->restriction[restrictionType].blocks[2][buffer], NULL, 12);}
  _timeEnd = CycleTime();
  level_f->cycles.restriction_unpack += (_timeEnd-_timeStart);
#endif

  level_f->cycles.restriction_total += (uint64_t)(CycleTime()-_timeCommunicationStart);
}
