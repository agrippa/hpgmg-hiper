//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------

extern mg_type *all_grids;

void cb_unpack_res(int srcid, int pos, int type, int depth_f, int id_c, int depth_c) {

  uint64_t _timeCommunicationStart = CycleTime();
  uint64_t _timeStart,_timeEnd;

  level_type *level_f;
  level_type *level_c;
  int buffer;
  double *buf;

  _timeStart = CycleTime();

  level_f = all_grids->levels[depth_f];
  level_c = all_grids->levels[depth_c];

  int i;
  size_t nth = MAX_NBGS*id_c;
  int *p = (int *) level_c->restriction[type].rflag;
  if (p[nth+pos] != 0) {
    printf("Wrong in Ping Res Handler Proc %d recv msg from %d for val %d\n", MYTHREAD, srcid, p[pos+nth]);
  }
  else {
    p[nth+pos] =1;
  }

  int bstart = level_c->restriction[type].sblock2[pos];
  int bend   = level_c->restriction[type].sblock2[pos+1];

  //  PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,bend-bstart)
  for(buffer=bstart;buffer<bend;buffer++){
    CopyBlock(level_c,id_c,&level_c->restriction[type].blocks[2][buffer]);
  }

  _timeEnd = CycleTime();
  level_c->cycles.restriction_unpack += (_timeEnd-_timeStart);

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
     box_type *lbox = (box_type *) block->read.boxgp;
     hclib::upcxx::global_ptr<double> gp = lbox->vectors[id_f] + lbox->ghosts*(1+lbox->jStride+lbox->kStride); 
     read = (double *)gp;
     read_jStride = lbox->jStride;
     read_kStride = lbox->kStride;
#else  // USE_UPCXX
     read = level_f->my_boxes[ block->read.box].vectors[id_f] + level_f->my_boxes[ block->read.box].ghosts*(1+level_f->my_boxes[ block->read.box].jStride+level_f->my_boxes[ block->read.box].kStride);
     read_jStride = level_f->my_boxes[block->read.box ].jStride;
     read_kStride = level_f->my_boxes[block->read.box ].kStride;
#endif
  }
  if(block->write.box>=0){
#ifdef USE_UPCXX
    box_type *lbox = (box_type *) block->write.boxgp;
    hclib::upcxx::global_ptr<double> gp = lbox->vectors[id_c] + lbox->ghosts*(1+lbox->jStride+lbox->kStride); 
    write = (double *)gp;
    write_jStride = lbox->jStride;
    write_kStride = lbox->kStride;
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

  // perform local restriction[restrictionType]... try and hide within Isend latency... 
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_f->restriction[restrictionType].num_blocks[3])
  for(buffer=0;buffer<level_f->restriction[restrictionType].num_blocks[3];buffer++){RestrictBlock(level_c,id_c,level_f,id_f,&level_f->restriction[restrictionType].blocks[3][buffer],restrictionType);}
  _timeEnd = CycleTime();
  level_f->cycles.restriction_shm += (_timeEnd-_timeStart);

  // pack MPI send buffers...
  _timeStart = CycleTime();
  PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_f->restriction[restrictionType].num_blocks[0])
  for(buffer=0;buffer<level_f->restriction[restrictionType].num_blocks[0];buffer++){RestrictBlock(level_c,id_c,level_f,id_f,&level_f->restriction[restrictionType].blocks[0][buffer],restrictionType);}
  _timeEnd = CycleTime();
  level_f->cycles.restriction_pack += (_timeEnd-_timeStart);

 
  // loop through MPI send buffers and post Isend's...
  _timeStart = CycleTime();
#ifdef USE_UPCXX
  int nshm = 0;
  for(n=0;n<level_f->restriction[restrictionType].num_sends;n++){
    hclib::upcxx::global_ptr<double> p1, p2;
    p1 = level_f->restriction[restrictionType].global_send_buffers[n];
    p2 = level_f->restriction[restrictionType].global_match_buffers[n];

    if (!hclib::upcxx::is_memory_shared_with(level_f->restriction[restrictionType].send_ranks[n])) {
      hclib::upcxx::event* copy_e = &level_f->restriction[restrictionType].copy_e[n];
      hclib::upcxx::async_copy(p1, p2, level_f->restriction[restrictionType].send_sizes[n], copy_e);
    } else {
      int rid = level_f->restriction[restrictionType].send_ranks[n];
      int pos = level_f->restriction[restrictionType].send_match_pos[n];
      size_t nth = MAX_NBGS*id_c;

      int *p = (int *) level_f->restriction[restrictionType].match_rflag[n]; *(p+nth+pos) = 1;
      nshm++;
    }

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
#ifdef USE_UPCXX

  for(n=0;n<level_f->restriction[restrictionType].num_sends;n++){
    int rid = level_f->restriction[restrictionType].send_ranks[n];

    if (!hclib::upcxx::is_memory_shared_with(rid)) {
      int cnt = level_f->restriction[restrictionType].send_sizes[n];
      int pos = level_f->restriction[restrictionType].send_match_pos[n];
      hclib::upcxx::event* copy_e = &level_f->restriction[restrictionType].copy_e[n];
      hclib::upcxx::event* data_e = &level_f->restriction[restrictionType].data_e[n];
      async_after(rid, copy_e, data_e)(cb_unpack_res, level_f->my_rank, pos, restrictionType, level_f->depth, id_c, level_c->depth);
    }
  }

  hclib::upcxx::async_wait();

  size_t nth = MAX_NBGS*id_c;

  if (level_c->restriction[restrictionType].num_recvs > 0) {
  int *p = (int *) level_c->restriction[restrictionType].rflag;

  while (1) {
    int arrived = 0;
    for (int n = 0; n < level_c->restriction[restrictionType].num_recvs; n++) {
      if (level_c->restriction[restrictionType].rflag[nth+n]==1) arrived++;
    }
    if (arrived == level_c->restriction[restrictionType].num_recvs) break;
    upcxx::advance();
  }
  for (int n = 0; n < level_c->restriction[restrictionType].num_recvs; n++) {
    p[nth+n] = 0;
  }

  }

#endif

  _timeEnd = CycleTime();
  level_f->cycles.restriction_wait += (_timeEnd-_timeStart);

  level_f->cycles.restriction_total += (uint64_t)(CycleTime()-_timeCommunicationStart);

}
