//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <stdint.h>
#include <gasnet.h>
#include <gasnet_tools.h>
uint64_t CycleTime(){
  return((uint64_t)(gasnett_ticks_to_ns(gasnett_ticks_now()))); // convert DP time in seconds to 64b integer nanosecond counter...
}
