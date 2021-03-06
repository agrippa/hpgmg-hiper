//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
void zero_vector(level_type * level, int component_id){
  // zero's the entire grid INCLUDING ghost zones...
  uint64_t _timeStart = CycleTime();
  int block;

  // PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
  hclib::future_t *fut = parallel_across_blocks(level, block, level->num_my_blocks,
          [&level, &component_id] (int block) {
    const int box = level->my_blocks[block].read.box;
          int ilo = level->my_blocks[block].read.i;
          int jlo = level->my_blocks[block].read.j;
          int klo = level->my_blocks[block].read.k;
          int ihi = level->my_blocks[block].dim.i + ilo;
          int jhi = level->my_blocks[block].dim.j + jlo;
          int khi = level->my_blocks[block].dim.k + klo;
    int i,j,k;

    box_type *lbox = (box_type *)&(level->my_boxes[box]);
    const int jStride = lbox->jStride;
    const int kStride = lbox->kStride;
    const int  ghosts = lbox->ghosts;
    const int     dim = lbox->dim;

    // expand the size of the block to include the ghost zones...
    if(ilo<=  0)ilo-=ghosts; 
    if(jlo<=  0)jlo-=ghosts; 
    if(klo<=  0)klo-=ghosts; 
    if(ihi>=dim)ihi+=ghosts; 
    if(jhi>=dim)jhi+=ghosts; 
    if(khi>=dim)khi+=ghosts; 

    double * __restrict__ grid = (double *)(lbox->vectors[component_id] + ghosts*(1+jStride+kStride));

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
      int ijk = i + j*jStride + k*kStride;
      grid[ijk] = 0.0;
    }}}
  });
  fut->wait();
  level->cycles.blas1 += (uint64_t)(CycleTime()-_timeStart);
}


//------------------------------------------------------------------------------------------------------------------------------
void initialize_valid_region(level_type * level){
  uint64_t _timeStart = CycleTime();
  int block;

  // PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
  hclib::future_t *fut = parallel_across_blocks(level, block, level->num_my_blocks,
          [level] (int block) {
    const int box = level->my_blocks[block].read.box;
          int ilo = level->my_blocks[block].read.i; 
          int jlo = level->my_blocks[block].read.j; 
          int klo = level->my_blocks[block].read.k; 
          int ihi = level->my_blocks[block].dim.i + ilo;
          int jhi = level->my_blocks[block].dim.j + jlo;
          int khi = level->my_blocks[block].dim.k + klo;
    int i,j,k;

    box_type *lbox = (box_type *)&(level->my_boxes[box]);
    const int jStride = lbox->jStride;
    const int kStride = lbox->kStride;
    const int  ghosts = lbox->ghosts;
    const int     dim = lbox->dim;

    // expand the size of the block to include the ghost zones...
    if(ilo<=  0)ilo-=ghosts; 
    if(jlo<=  0)jlo-=ghosts; 
    if(klo<=  0)klo-=ghosts; 
    if(ihi>=dim)ihi+=ghosts; 
    if(jhi>=dim)jhi+=ghosts; 
    if(khi>=dim)khi+=ghosts; 

    double * __restrict__ valid = (double *)(lbox->vectors[VECTOR_VALID] + ghosts*(1+jStride+kStride));

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
      int ijk = i + j*jStride + k*kStride;
      valid[ijk] = 1.0; // i.e. all cells including ghosts are valid for periodic BC's
      if(level->boundary_condition.type == BC_DIRICHLET){ // cells outside the domain boundaries are not valid

        if(i + lbox->low.i <             0)valid[ijk] = 0.0;
        if(j + lbox->low.j <             0)valid[ijk] = 0.0;
        if(k + lbox->low.k <             0)valid[ijk] = 0.0;
        if(i + lbox->low.i >= level->dim.i)valid[ijk] = 0.0;
        if(j + lbox->low.j >= level->dim.j)valid[ijk] = 0.0;
        if(k + lbox->low.k >= level->dim.k)valid[ijk] = 0.0;

      }
    }}}
  }); // });
  fut->wait();
  level->cycles.blas1 += (uint64_t)(CycleTime()-_timeStart);
}


//------------------------------------------------------------------------------------------------------------------------------
void initialize_grid_to_scalar(level_type * level, int component_id, double scalar){
  // initializes the grid to a scalar while zero'ing the ghost zones...
  uint64_t _timeStart = CycleTime();
  int block;

  // PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
  hclib::future_t *fut = parallel_across_blocks(level, block, level->num_my_blocks,
          [&level, &component_id, &scalar] (int block) {
    const int box = level->my_blocks[block].read.box;
          int ilo = level->my_blocks[block].read.i;
          int jlo = level->my_blocks[block].read.j;
          int klo = level->my_blocks[block].read.k;
          int ihi = level->my_blocks[block].dim.i + ilo;
          int jhi = level->my_blocks[block].dim.j + jlo;
          int khi = level->my_blocks[block].dim.k + klo;
    int i,j,k;

    box_type *lbox = (box_type *)&(level->my_boxes[box]);
    const int jStride = lbox->jStride;
    const int kStride = lbox->kStride;
    const int  ghosts = lbox->ghosts;
    const int     dim = lbox->dim;

    // expand the size of the block to include the ghost zones...
    if(ilo<=  0)ilo-=ghosts; 
    if(jlo<=  0)jlo-=ghosts; 
    if(klo<=  0)klo-=ghosts; 
    if(ihi>=dim)ihi+=ghosts; 
    if(jhi>=dim)jhi+=ghosts; 
    if(khi>=dim)khi+=ghosts; 

    double * __restrict__ grid = (double *)(lbox->vectors[component_id] + ghosts*(1+jStride+kStride));

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
        int ijk = i + j*jStride + k*kStride;
        int ghostZone = (i<0) || (j<0) || (k<0) || (i>=dim) || (j>=dim) || (k>=dim);
        grid[ijk] = ghostZone ? 0.0 : scalar;
    }}}
  });
  fut->wait();
  level->cycles.blas1 += (uint64_t)(CycleTime()-_timeStart);
}


//------------------------------------------------------------------------------------------------------------------------------
void add_vectors(level_type * level, int id_c, double scale_a, int id_a, double scale_b, int id_b){ // c=scale_a*id_a + scale_b*id_b
  uint64_t _timeStart = CycleTime();

  int block;

  // PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
  hclib::future_t *fut = parallel_across_blocks(level, block, level->num_my_blocks,
          [&level, &id_c, &id_a, &id_b, &scale_a, &scale_b] (int block) {
    const int box = level->my_blocks[block].read.box;
    const int ilo = level->my_blocks[block].read.i;
    const int jlo = level->my_blocks[block].read.j;
    const int klo = level->my_blocks[block].read.k;
    const int ihi = level->my_blocks[block].dim.i + ilo;
    const int jhi = level->my_blocks[block].dim.j + jlo;
    const int khi = level->my_blocks[block].dim.k + klo;
    int i,j,k;

    box_type *lbox = (box_type *)&(level->my_boxes[box]);
    const int jStride = lbox->jStride;
    const int kStride = lbox->kStride;
    const int  ghosts = lbox->ghosts;
    const int     dim = lbox->dim;

    double * __restrict__ grid_c = (double *)(lbox->vectors[id_c] + ghosts*(1+jStride+kStride));
    double * __restrict__ grid_a = (double *)(lbox->vectors[id_a] + ghosts*(1+jStride+kStride));
    double * __restrict__ grid_b = (double *)(lbox->vectors[id_b] + ghosts*(1+jStride+kStride));

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
        int ijk = i + j*jStride + k*kStride;
        grid_c[ijk] = scale_a*grid_a[ijk] + scale_b*grid_b[ijk];
    }}}
  });
  fut->wait();
  level->cycles.blas1 += (uint64_t)(CycleTime()-_timeStart);
}


//------------------------------------------------------------------------------------------------------------------------------
void mul_vectors(level_type * level, int id_c, double scale, int id_a, int id_b){ // id_c=scale*id_a*id_b
  uint64_t _timeStart = CycleTime();

  int block;

  // PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
  hclib::future_t *fut = parallel_across_blocks(level, block, level->num_my_blocks,
          [&level, &id_c, &id_a, &id_b, &scale] (int block) {
    const int box = level->my_blocks[block].read.box;
    const int ilo = level->my_blocks[block].read.i;
    const int jlo = level->my_blocks[block].read.j;
    const int klo = level->my_blocks[block].read.k;
    const int ihi = level->my_blocks[block].dim.i + ilo;
    const int jhi = level->my_blocks[block].dim.j + jlo;
    const int khi = level->my_blocks[block].dim.k + klo;
    int i,j,k;

    box_type *lbox = (box_type *)&(level->my_boxes[box]);
    const int jStride = lbox->jStride;
    const int kStride = lbox->kStride;
    const int  ghosts = lbox->ghosts;

    double * __restrict__ grid_c = (double *)(lbox->vectors[id_c] + ghosts*(1+jStride+kStride));
    double * __restrict__ grid_a = (double *)(lbox->vectors[id_a] + ghosts*(1+jStride+kStride));
    double * __restrict__ grid_b = (double *)(lbox->vectors[id_b] + ghosts*(1+jStride+kStride));

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
        int ijk = i + j*jStride + k*kStride;
        grid_c[ijk] = scale*grid_a[ijk]*grid_b[ijk];
    }}}
  });
  fut->wait();
  level->cycles.blas1 += (uint64_t)(CycleTime()-_timeStart);
}


//------------------------------------------------------------------------------------------------------------------------------
void invert_vector(level_type * level, int id_c, double scale_a, int id_a){ // c[]=scale_a/a[]
  uint64_t _timeStart = CycleTime();

  int block;

  // PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
  hclib::future_t *fut = parallel_across_blocks(level, block, level->num_my_blocks,
          [&level, &id_c, &id_a, &scale_a] (int block) {
    const int box = level->my_blocks[block].read.box;
    const int ilo = level->my_blocks[block].read.i;
    const int jlo = level->my_blocks[block].read.j;
    const int klo = level->my_blocks[block].read.k;
    const int ihi = level->my_blocks[block].dim.i + ilo;
    const int jhi = level->my_blocks[block].dim.j + jlo;
    const int khi = level->my_blocks[block].dim.k + klo;
    int i,j,k;

    box_type *lbox = (box_type *)&(level->my_boxes[box]);
    const int jStride = lbox->jStride;
    const int kStride = lbox->kStride;
    const int  ghosts = lbox->ghosts;

    double * __restrict__ grid_c = (double *)(lbox->vectors[id_c] + ghosts*(1+jStride+kStride));
    double * __restrict__ grid_a = (double *)(lbox->vectors[id_a] + ghosts*(1+jStride+kStride));

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
        int ijk = i + j*jStride + k*kStride;
        grid_c[ijk] = scale_a/grid_a[ijk];
    }}}
  });
  fut->wait();
  level->cycles.blas1 += (uint64_t)(CycleTime()-_timeStart);
}


//------------------------------------------------------------------------------------------------------------------------------
void scale_vector(level_type * level, int id_c, double scale_a, int id_a){ // c[]=scale_a*a[]
  uint64_t _timeStart = CycleTime();

  int block;

  // PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
  hclib::future_t *fut = parallel_across_blocks(level, block, level->num_my_blocks,
          [&level, &id_c, &id_a, &scale_a] (int block) {
    const int box = level->my_blocks[block].read.box;
    const int ilo = level->my_blocks[block].read.i;
    const int jlo = level->my_blocks[block].read.j;
    const int klo = level->my_blocks[block].read.k;
    const int ihi = level->my_blocks[block].dim.i + ilo;
    const int jhi = level->my_blocks[block].dim.j + jlo;
    const int khi = level->my_blocks[block].dim.k + klo;
    int i,j,k;

    box_type *lbox = (box_type *)&(level->my_boxes[box]);
    const int jStride = lbox->jStride;
    const int kStride = lbox->kStride;
    const int  ghosts = lbox->ghosts;

    double * __restrict__ grid_c = (double *)(lbox->vectors[id_c] + ghosts*(1+jStride+kStride));
    double * __restrict__ grid_a = (double *)(lbox->vectors[id_a] + ghosts*(1+jStride+kStride));

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
        int ijk = i + j*jStride + k*kStride;
        grid_c[ijk] = scale_a*grid_a[ijk];
    }}}
  });
  fut->wait();
  level->cycles.blas1 += (uint64_t)(CycleTime()-_timeStart);
}


//------------------------------------------------------------------------------------------------------------------------------
double dot(level_type * level, int id_a, int id_b){
  uint64_t _timeStart = CycleTime();


  int block;
  hclib::atomic_sum_t<double> a_dot_b_level_atomic(0.0);

  // PRAGMA_THREAD_ACROSS_BLOCKS_SUM(level,block,level->num_my_blocks,a_dot_b_level)
  hclib::future_t *fut = parallel_across_blocks(level, block, level->num_my_blocks,
          [&level, &id_a, &id_b, &a_dot_b_level_atomic] (int block) {
    const int box = level->my_blocks[block].read.box;
    const int ilo = level->my_blocks[block].read.i;
    const int jlo = level->my_blocks[block].read.j;
    const int klo = level->my_blocks[block].read.k;
    const int ihi = level->my_blocks[block].dim.i + ilo;
    const int jhi = level->my_blocks[block].dim.j + jlo;
    const int khi = level->my_blocks[block].dim.k + klo;
    int i,j,k;

    box_type *lbox = (box_type *)&(level->my_boxes[box]);
    const int jStride = lbox->jStride;
    const int kStride = lbox->kStride;
    const int  ghosts = lbox->ghosts;

    double * __restrict__ grid_a = (double *)(lbox->vectors[id_a] + ghosts*(1+jStride+kStride));
    double * __restrict__ grid_b = (double *)(lbox->vectors[id_b] + ghosts*(1+jStride+kStride));

    double a_dot_b_block = 0.0;

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
      int ijk = i + j*jStride + k*kStride;
      a_dot_b_block += grid_a[ijk]*grid_b[ijk];
    }}}
    a_dot_b_level_atomic+=a_dot_b_block;
  });
  fut->wait();
  level->cycles.blas1 += (uint64_t)(CycleTime()-_timeStart);

  double a_dot_b_level = a_dot_b_level_atomic.get();

  #ifdef USE_MPI
  uint64_t _timeStartAllReduce = CycleTime();
  double send = a_dot_b_level;
  hclib::MPI_Allreduce(&send,&a_dot_b_level,1,MPI_DOUBLE,MPI_SUM,level->MPI_COMM_ALLREDUCE);
  uint64_t _timeEndAllReduce = CycleTime();
  level->cycles.collectives   += (uint64_t)(_timeEndAllReduce-_timeStartAllReduce);
  #endif

  return(a_dot_b_level);
}

//------------------------------------------------------------------------------------------------------------------------------
double norm(level_type * level, int component_id){ // implements the max norm
  uint64_t _timeStart = CycleTime();

  int block;
  hclib::atomic_max_t<double> max_norm_atomic(0.0);


  // PRAGMA_THREAD_ACROSS_BLOCKS_MAX(level,block,level->num_my_blocks,max_norm)
  hclib::future_t *fut = parallel_across_blocks(level, block, level->num_my_blocks,
          [&level, &component_id, &max_norm_atomic] (int block) {
    const int box = level->my_blocks[block].read.box;
    const int ilo = level->my_blocks[block].read.i;
    const int jlo = level->my_blocks[block].read.j;
    const int klo = level->my_blocks[block].read.k;
    const int ihi = level->my_blocks[block].dim.i + ilo;
    const int jhi = level->my_blocks[block].dim.j + jlo;
    const int khi = level->my_blocks[block].dim.k + klo;
    int i,j,k;

    box_type *lbox = (box_type *)&(level->my_boxes[box]);
    const int jStride = lbox->jStride;
    const int kStride = lbox->kStride;
    const int  ghosts = lbox->ghosts;
    double * __restrict__ grid = (double *)(lbox->vectors[component_id] + ghosts*(1+jStride+kStride));

    double block_norm = 0.0;

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){ 
      int ijk = i + j*jStride + k*kStride;
      double fabs_grid_ijk = fabs(grid[ijk]);
      if(fabs_grid_ijk>block_norm){block_norm=fabs_grid_ijk;} // max norm
    }}}

    max_norm_atomic.update(block_norm);
  }); // block list
  fut->wait();
  level->cycles.blas1 += (uint64_t)(CycleTime()-_timeStart);

  double max_norm = max_norm_atomic.get();

  #ifdef USE_MPI
  uint64_t _timeStartAllReduce = CycleTime();
  double send = max_norm;
  hclib::MPI_Allreduce(&send,&max_norm,1,MPI_DOUBLE,MPI_MAX,level->MPI_COMM_ALLREDUCE);
  uint64_t _timeEndAllReduce = CycleTime();
  level->cycles.collectives   += (uint64_t)(_timeEndAllReduce-_timeStartAllReduce);
  #endif
  return(max_norm);
}


//------------------------------------------------------------------------------------------------------------------------------
double mean(level_type * level, int id_a){
  uint64_t _timeStart = CycleTime();


  int block;
  hclib::atomic_sum_t<double> sum_level_atomic(0.0);

  // PRAGMA_THREAD_ACROSS_BLOCKS_SUM(level,block,level->num_my_blocks,sum_level)
  hclib::future_t *fut = parallel_across_blocks(level, block, level->num_my_blocks,
          [&level, &id_a, &sum_level_atomic] (int block) {
    const int box = level->my_blocks[block].read.box;
    const int ilo = level->my_blocks[block].read.i;
    const int jlo = level->my_blocks[block].read.j;
    const int klo = level->my_blocks[block].read.k;
    const int ihi = level->my_blocks[block].dim.i + ilo;
    const int jhi = level->my_blocks[block].dim.j + jlo;
    const int khi = level->my_blocks[block].dim.k + klo;
    int i,j,k;

    box_type *lbox = (box_type *)&(level->my_boxes[box]);
    int jStride = lbox->jStride;
    const int kStride = lbox->kStride;
    const int  ghosts = lbox->ghosts;
    double * __restrict__ grid_a = (double *)(lbox->vectors[id_a] + ghosts*(1+jStride+kStride));

    double sum_block = 0.0;

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
      int ijk = i + j*jStride + k*kStride;
      sum_block += grid_a[ijk];
    }}}
    sum_level_atomic += sum_block;
  });
  fut->wait();
  level->cycles.blas1 += (uint64_t)(CycleTime()-_timeStart);
  double ncells_level = (double)level->dim.i*(double)level->dim.j*(double)level->dim.k;

  double sum_level = sum_level_atomic.get();

  #ifdef USE_MPI
  uint64_t _timeStartAllReduce = CycleTime();
  double send = sum_level;
  hclib::MPI_Allreduce(&send,&sum_level,1,MPI_DOUBLE,MPI_SUM,level->MPI_COMM_ALLREDUCE);
  uint64_t _timeEndAllReduce = CycleTime();
  level->cycles.collectives   += (uint64_t)(_timeEndAllReduce-_timeStartAllReduce);
  #endif

  double mean_level = sum_level / ncells_level;
  return(mean_level);
}


//------------------------------------------------------------------------------------------------------------------------------
void shift_vector(level_type * level, int id_c, int id_a, double shift_a){
  uint64_t _timeStart = CycleTime();
  int block;

  // PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
  hclib::future_t *fut = parallel_across_blocks(level, block, level->num_my_blocks,
          [&level, &id_c, &id_a, &shift_a] (int block) {
    const int box = level->my_blocks[block].read.box;
    const int ilo = level->my_blocks[block].read.i;
    const int jlo = level->my_blocks[block].read.j;
    const int klo = level->my_blocks[block].read.k;
    const int ihi = level->my_blocks[block].dim.i + ilo;
    const int jhi = level->my_blocks[block].dim.j + jlo;
    const int khi = level->my_blocks[block].dim.k + klo;
    int i,j,k;

    box_type *lbox = (box_type *)&(level->my_boxes[box]);
    const int jStride = lbox->jStride;
    const int kStride = lbox->kStride;
    const int  ghosts = lbox->ghosts;
    double * __restrict__ grid_c = (double *)(lbox->vectors[id_c] + ghosts*(1+jStride+kStride));
    double * __restrict__ grid_a = (double *)(lbox->vectors[id_a] + ghosts*(1+jStride+kStride));

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
      int ijk = i + j*jStride + k*kStride;
      grid_c[ijk] = grid_a[ijk] + shift_a;
    }}}
  });
  fut->wait();
  level->cycles.blas1 += (uint64_t)(CycleTime()-_timeStart);
}

//------------------------------------------------------------------------------------------------------------------------------
void project_cell_to_face(level_type * level, int id_cell, int id_face, int dir){
  uint64_t _timeStart = CycleTime();
  int block;

  // PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
  hclib::future_t *fut = parallel_across_blocks(level, block, level->num_my_blocks,
          [&level, &id_cell, &id_face, &dir] (int block) {
    const int box = level->my_blocks[block].read.box;
    const int ilo = level->my_blocks[block].read.i;
    const int jlo = level->my_blocks[block].read.j;
    const int klo = level->my_blocks[block].read.k;
    const int ihi = level->my_blocks[block].dim.i + ilo;
    const int jhi = level->my_blocks[block].dim.j + jlo;
    const int khi = level->my_blocks[block].dim.k + klo;
    int i,j,k;

    box_type *lbox = (box_type *)&(level->my_boxes[box]);
    const int jStride = lbox->jStride;
    const int kStride = lbox->kStride;
    const int  ghosts = lbox->ghosts;
    double * __restrict__ grid_cell = (double *)(lbox->vectors[id_cell] + ghosts*(1+jStride+kStride));
    double * __restrict__ grid_face = (double *)(lbox->vectors[id_face] + ghosts*(1+jStride+kStride));

    int stride;
    switch(dir){
      case 0: stride =       1;break;//i-direction
      case 1: stride = jStride;break;//j-direction
      case 2: stride = kStride;break;//k-direction
    }

    for(k=klo;k<=khi;k++){ // <= to ensure you do low and high faces
    for(j=jlo;j<=jhi;j++){
    for(i=ilo;i<=ihi;i++){
      int ijk = i + j*jStride + k*kStride;
      grid_face[ijk] = 0.5*(grid_cell[ijk-stride] + grid_cell[ijk]); // simple linear interpolation
    }}}
  });
  fut->wait();

  level->cycles.blas1 += (uint64_t)(CycleTime()-_timeStart);
}


//------------------------------------------------------------------------------------------------------------------------------
double error(level_type * level, int id_a, int id_b){
  double h3 = level->h * level->h * level->h;
               add_vectors(level,VECTOR_TEMP,1.0,id_a,-1.0,id_b);            // VECTOR_TEMP = id_a - id_b
  double   max =      norm(level,VECTOR_TEMP);                return(max);   // max norm of error function
  double    L2 = sqrt( dot(level,VECTOR_TEMP,VECTOR_TEMP)*h3);return( L2);   // normalized L2 error ?
}
