//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
//#define GSRB_STRIDE2
//#define GSRB_FP
//------------------------------------------------------------------------------------------------------------------------------
void smooth(level_type * level, int phi_id, int rhs_id, double a, double b){
  int block,s;
  for(s=0;s<2*NUM_SMOOTHS;s++){ // there are two sweeps per GSRB smooth
    // exchange the ghost zone...
    exchange_boundary(level,phi_id,stencil_is_star_shaped());apply_BCs(level,phi_id,stencil_is_star_shaped());

    // apply the smoother...
    uint64_t _timeStart = CycleTime();

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
      const int ghosts = level->box_ghosts;

      box_type *lbox = &(level->my_boxes[box]);
      const int color000 = (lbox->low.i^lbox->low.j^lbox->low.k)&1;  // is element 000 red or black ???  (should only be an issue if box dimension is odd)
      const int jStride = lbox->jStride;
      const int kStride = lbox->kStride;
      const int     dim = lbox->dim;
      const double h2inv = 1.0/(level->h*level->h);
      const double * __restrict__ phi      = lbox->vectors[       phi_id] + ghosts*(1+jStride+kStride); // i.e. [0] = first non ghost zone point
            double * __restrict__ phi_new  = lbox->vectors[       phi_id] + ghosts*(1+jStride+kStride); // i.e. [0] = first non ghost zone point
      const double * __restrict__ rhs      = lbox->vectors[       rhs_id] + ghosts*(1+jStride+kStride);
      const double * __restrict__ alpha    = lbox->vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride);
      const double * __restrict__ beta_i   = lbox->vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride);
      const double * __restrict__ beta_j   = lbox->vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride);
      const double * __restrict__ beta_k   = lbox->vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride);
      const double * __restrict__ Dinv     = lbox->vectors[VECTOR_DINV  ] + ghosts*(1+jStride+kStride);
      const double * __restrict__ valid    = lbox->vectors[VECTOR_VALID ] + ghosts*(1+jStride+kStride); // cell is inside the domain

      const double * __restrict__ RedBlack[2] = {level->RedBlack_FP[0] + ghosts*(1+jStride), 
                                                 level->RedBlack_FP[1] + ghosts*(1+jStride)};
          

      #if defined(GSRB_FP)
      #warning GSRB using pre-computed 1.0/0.0 FP array for Red-Black to facilitate vectorization...
      for(k=klo;k<khi;k++){
      for(j=jlo;j<jhi;j++){
      for(i=ilo;i<ihi;i++){
            int EvenOdd = (k^s^color000)&1;
            int ij  = i + j*jStride;
            int ijk = i + j*jStride + k*kStride;
            double Ax     = apply_op_ijk(phi);
            double lambda =     Dinv_ijk();
            phi_new[ijk] = phi[ijk] + RedBlack[EvenOdd][ij]*lambda*(rhs[ijk]-Ax); // compiler seems to get confused unless there are disjoint read/write pointers
      }}}
      #elif defined(GSRB_STRIDE2)
      #warning GSRB using stride-2 accesses to minimie the number of flop's
      #error verify this still works...
      for(k=klo;k<khi;k++){
      for(j=jlo;j<jhi;j++){
      for(i=ilo+((j^k^s^color000)&1)+1-ghosts;i<ihi;i+=2){ // stride-2 GSRB
            int ijk = i + j*jStride + k*kStride; 
            double Ax     = apply_op_ijk(phi);
            double lambda =     Dinv_ijk();
            phi_new[ijk] = phi[ijk] + lambda*(rhs[ijk]-Ax);
      }}}
      #else
      #warning GSRB using if-then-else on loop indices for Red-Black because its easy to read...
      for(k=klo;k<khi;k++){
      for(j=jlo;j<jhi;j++){
      for(i=ilo;i<ihi;i++){
      if((i^j^k^s^color000^1)&1){ // looks very clean when [0] is i,j,k=0,0,0 
            int ijk = i + j*jStride + k*kStride;
            double Ax     = apply_op_ijk(phi);
            double lambda =     Dinv_ijk();
            phi_new[ijk] = phi[ijk] + lambda*(rhs[ijk]-Ax);
      }}}}
      #endif
    } // boxes
    level->cycles.smooth += (uint64_t)(CycleTime()-_timeStart);
  } // s-loop
}


//------------------------------------------------------------------------------------------------------------------------------
