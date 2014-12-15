//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
// shan: flag = 0, the same with original; flag =1 , use src as read buffer
static inline void CopyBlock(level_type *level, int id, blockCopy_type *block, double *src, int flag){
  // copy 3D array from read_i,j,k of read[] to write_i,j,k in write[]
  int   dim_i       = block->dim.i;
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

  if (flag == 1) read = src;

#ifdef USE_UPCXX
  if(block->read.box >=0) {
#ifdef UPCXX_SHARED
    int rank = level->rank_of_box(block->read.box);
    if (!upcxx::is_memory_shared_with(rank)) {
      printf("Wrong: Proc %d level %d read box %d rank is %d not shared!\n", level->my_rank, level->depth, block->read.box, rank);
      exit(1);
    }
    global_ptr<box_type> box = level->addr_of_box(block->read.box);
    box_type *lbox = (box_type *)box;
#else
    box_type *lbox = &(level->my_boxes[block->read.box]);
#endif
    read = lbox->vectors[id] + lbox->ghosts*(1+lbox->jStride+lbox->kStride);
  }
  if(block->write.box>=0) {
#ifdef UPCXX_SHARED
    int rank = level->rank_of_box(block->write.box);
    if (!upcxx::is_memory_shared_with(rank)) {
      printf("Wrong: Proc %d level %d write box %d rank is %d not shared!\n", level->my_rank, level->depth, block->write.box, rank);
      exit(1);
    }
    global_ptr<box_type> box = level->addr_of_box(block->write.box);
    box_type *lbox = (box_type *)box;
#else
    box_type *lbox = &(level->my_boxes[block->write.box]);
#endif
    write = lbox->vectors[id] + lbox->ghosts*(1+lbox->jStride+lbox->kStride);
  }
#else
  if(block->read.box >=0) read = level->my_boxes[ block->read.box].vectors[id] + level->my_boxes[ block->read.box].ghosts*(1+level->my_boxes[ block->read.box].jStride+level->my_boxes[ block->read.box].kStride);
  if(block->write.box>=0)write = level->my_boxes[block->write.box].vectors[id] + level->my_boxes[block->write.box].ghosts*(1+level->my_boxes[block->write.box].jStride+level->my_boxes[block->write.box].kStride);
#endif

  int i,j,k;
  if(dim_i==1){ // be smart and don't have an inner loop from 0 to 0
    for(k=0;k<dim_k;k++){
    for(j=0;j<dim_j;j++){
      int  read_ijk = ( read_i) + (j+ read_j)* read_jStride + (k+ read_k)* read_kStride;
      int write_ijk = (write_i) + (j+write_j)*write_jStride + (k+write_k)*write_kStride;
      write[write_ijk] = read[read_ijk];
    }}
  }else if(dim_j==1){ // don't have a 0..0 loop
    for(k=0;k<dim_k;k++){
    for(i=0;i<dim_i;i++){
      int  read_ijk = (i+ read_i) + ( read_j)* read_jStride + (k+ read_k)* read_kStride;
      int write_ijk = (i+write_i) + (write_j)*write_jStride + (k+write_k)*write_kStride;
      write[write_ijk] = read[read_ijk];
    }}
  }else if(dim_k==1){ // don't have a 0..0 loop
    for(j=0;j<dim_j;j++){
    for(i=0;i<dim_i;i++){
      int  read_ijk = (i+ read_i) + (j+ read_j)* read_jStride + ( read_k)* read_kStride;
      int write_ijk = (i+write_i) + (j+write_j)*write_jStride + (write_k)*write_kStride;
      write[write_ijk] = read[read_ijk];
    }}
  }else if(dim_i==4){ // be smart and don't have an inner loop from 0 to 3
    for(k=0;k<dim_k;k++){
    for(j=0;j<dim_j;j++){
      int  read_ijk = ( read_i) + (j+ read_j)* read_jStride + (k+ read_k)* read_kStride;
      int write_ijk = (write_i) + (j+write_j)*write_jStride + (k+write_k)*write_kStride;
      write[write_ijk+0] = read[read_ijk+0];
      write[write_ijk+1] = read[read_ijk+1];
      write[write_ijk+2] = read[read_ijk+2];
      write[write_ijk+3] = read[read_ijk+3];
    }}
  }else{
    for(k=0;k<dim_k;k++){
    for(j=0;j<dim_j;j++){
    for(i=0;i<dim_i;i++){
      int  read_ijk = (i+ read_i) + (j+ read_j)* read_jStride + (k+ read_k)* read_kStride;
      int write_ijk = (i+write_i) + (j+write_j)*write_jStride + (k+write_k)*write_kStride;
      write[write_ijk] = read[read_ijk];
    }}}
  }

}


//------------------------------------------------------------------------------------------------------------------------------
static inline void IncrementBlock(level_type *level, int id, double prescale, blockCopy_type *block, double *src, int flag ){
  // copy 3D array from read_i,j,k of read[] to write_i,j,k in write[]
  int   dim_i       = block->dim.i;
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

  if (flag == 1) read = src;

  if(block->read.box >=0){
#ifdef USE_UPCXX
#ifdef UPCXX_SHARED
     int rank = level->rank_of_box(block->read.box);
     if (!upcxx::is_memory_shared_with(rank)) {
      printf("Wrong1: Proc %d level %d read box %d rank is %d not shared!\n", level->my_rank, level->depth, block->read.box, rank);
      exit(1);
     }
     global_ptr<box_type> box = level->addr_of_box(block->read.box);
     box_type *lbox = (box_type *)box;
#else
     box_type *lbox = &(level->my_boxes[block->read.box]);
#endif
     read = lbox->vectors[id] + lbox->ghosts*(1+lbox->jStride+lbox->kStride);
     read_jStride = lbox->jStride;
     read_kStride = lbox->kStride;
#else
     read = level->my_boxes[ block->read.box].vectors[id] + level->my_boxes[ block->read.box].ghosts*(1+level->my_boxes[ block->read.box].jStride+level->my_boxes[ block->read.box].kStride);
     read_jStride = level->my_boxes[block->read.box ].jStride;
     read_kStride = level->my_boxes[block->read.box ].kStride;
#endif
  }
  if(block->write.box>=0){
#ifdef USE_UPCXX
#ifdef UPCXX_SHARED
    int rank = level->rank_of_box(block->write.box);
    if (!upcxx::is_memory_shared_with(rank)) {
      printf("Wrong: Proc %d level %d write box %d rank is %d not shared!\n", level->my_rank, level->depth, block->write.box, rank);
      exit(1);
    }
    global_ptr<box_type> box = level->addr_of_box(block->write.box);
    box_type *lbox = (box_type *)box;
#else
    box_type *lbox = &(level->my_boxes[block->write.box]);
#endif
    write = lbox->vectors[id] + lbox->ghosts*(1+lbox->jStride+lbox->kStride);
    write_jStride = lbox->jStride;
    write_kStride = lbox->kStride;
#else
    write = level->my_boxes[block->write.box].vectors[id] + level->my_boxes[block->write.box].ghosts*(1+level->my_boxes[block->write.box].jStride+level->my_boxes[block->write.box].kStride);
    write_jStride = level->my_boxes[block->write.box].jStride;
    write_kStride = level->my_boxes[block->write.box].kStride;
#endif
  }

  int i,j,k;
  for(k=0;k<dim_k;k++){
  for(j=0;j<dim_j;j++){
  for(i=0;i<dim_i;i++){
    int  read_ijk = (i+ read_i) + (j+ read_j)* read_jStride + (k+ read_k)* read_kStride;
    int write_ijk = (i+write_i) + (j+write_j)*write_jStride + (k+write_k)*write_kStride;
    write[write_ijk] = prescale*write[write_ijk] + read[read_ijk]; // CAREFUL !!!  you must guarantee you zero'd the MPI buffers(write[]) and destination boxes at some point to avoid 0.0*NaN or 0.0*inf
  }}}

}

//------------------------------------------------------------------------------------------------------------------------------
