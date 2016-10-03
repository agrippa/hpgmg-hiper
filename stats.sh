#!/bin/bash

LBL=MGBuild
NRUNS=10

for NODES in 64 128 256 512; do
    MEAN_VANILLA=$(cat ${NODES}.out | grep 'Min Total' | grep $LBL | awk '{ print $7 }' | tail -n $NRUNS | python3 ~/mean.py)
    STD_VANILLA=$(cat ${NODES}.out | grep 'Min Total' | grep $LBL | awk '{ print $7 }' | tail -n $NRUNS | python3 ~/std.py)

    MEAN_HIPER=$(cat ${NODES}.out | grep 'Min Total' | grep $LBL | awk '{ print $7 }' | head -n $NRUNS | python3 ~/mean.py)
    STD_HIPER=$(cat ${NODES}.out | grep 'Min Total' | grep $LBL | awk '{ print $7 }' | head -n $NRUNS | python3 ~/std.py)

    echo -n $MEAN_VANILLA,$(echo $MEAN_VANILLA + \( 1.96 \* $STD_VANILLA \) | bc -l),$(echo $MEAN_VANILLA - \( 1.96 \* $STD_VANILLA \) | bc -l)
    echo ,$MEAN_HIPER,$(echo $MEAN_HIPER + \( 1.96 \* $STD_HIPER \) | bc -l),$(echo $MEAN_HIPER - \( 1.96 \* $STD_HIPER \) | bc -l)
done
