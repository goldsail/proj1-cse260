#!/bin/bash
AVX=3x12
for L1M in 18 21 24 27 30 36
do
for L1N in 24 36
do
for L1K in 24 28 32
do
for L2M in $(( 2 * L1M )) $(( 3 * L1M ))
do
for L2N in $(( 2 * L1N )) $(( 3 * L1N ))
do
for L2K in $(( 3 * L1K ))
do
L3M=$(( 8 * L2M ))
L3N=$(( 8 * L2N ))
L3K=$(( 8 * L2K ))
echo "nonsq-$AVX-$L1M-$L1N-$L1K-$L2M-$L2N-$L2K-$L3M-$L3N-$L3K.txt"
make MY_OPT="-O4 -DSIMD_$AVX -DCACHELAYOUT -DL1_BLOCK_SIZE_M=$L1M -DL1_BLOCK_SIZE_N=$L1N -DL1_BLOCK_SIZE_K=$L1K -DL2_BLOCK_SIZE_M=$L2M -DL2_BLOCK_SIZE_N=$L2N -DL2_BLOCK_SIZE_K=$L2K -DL3_BLOCK_SIZE_M=$L3M -DL3_BLOCK_SIZE_N=$L3N -DL3_BLOCK_SIZE_K=$L3K" 1>/dev/null 2>&1
./benchmark-blocked >tune/nonsq-$AVX-$L1M-$L1N-$L1K-$L2M-$L2N-$L2K-$L3M-$L3N-$L3K.txt
done
done
done
done
done
done
