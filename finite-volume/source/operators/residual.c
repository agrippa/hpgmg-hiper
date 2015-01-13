//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
// calculate res_id = rhs_id - A(x_id)

void residual(level_type * level, int res_id, int x_id, int rhs_id, double a, double b){
  // exchange the boundary for x in prep for Ax...
  exchange_boundary(level,x_id,stencil_is_star_shaped());
          apply_BCs(level,x_id,stencil_is_star_shaped());

  // now do residual/restriction proper...
  uint64_t _timeStart = CycleTime();
  int block;

  PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
  for(block=0;block<level->num_my_blocks;block++){
    const int box = level->my_blocks[block].read.box;
    const int ilo = level->my_blocks[block].read.i;
    const int jlo = level->my_blocks[block].read.j;
    const int klo = level->my_blocks[block].read.k;
    const int ihi = level->my_blocks[block].dim.i + ilo;
    const int jhi = level->my_blocks[block].dim.j + jlo;
    const int khi = level->my_blocks[block].dim.k + klo;
    int i,j,k;

      box_type *lbox = &(level->my_boxes[box]);
      const int jStride = lbox->jStride;
      const int kStride = lbox->kStride;
      const int  ghosts = lbox->ghosts;
      const int     dim = lbox->dim;
      const double h2inv = 1.0/(level->h*level->h);
      const double * __restrict__ x      = lbox->vectors[         x_id] + ghosts*(1+jStride+kStride); // i.e. [0] = first non ghost zone point
      const double * __restrict__ rhs    = lbox->vectors[       rhs_id] + ghosts*(1+jStride+kStride);
      const double * __restrict__ alpha  = lbox->vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride);
      const double * __restrict__ beta_i = lbox->vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride);
      const double * __restrict__ beta_j = lbox->vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride);
      const double * __restrict__ beta_k = lbox->vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride);
      const double * __restrict__ valid  = lbox->vectors[VECTOR_VALID ] + ghosts*(1+jStride+kStride); // cell is inside the domain
      double * __restrict__ res    = lbox->vectors[       res_id] + ghosts*(1+jStride+kStride);

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
      int ijk = i + j*jStride + k*kStride;
      double Ax = apply_op_ijk(x);
      res[ijk] = rhs[ijk]-Ax;
    }}}
  }
  level->cycles.residual += (uint64_t)(CycleTime()-_timeStart);
}

