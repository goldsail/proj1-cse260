#!/bin/bash
AVX=3x12_D
for L1M in 30
do
for L1N in 36
do
for L1K in 32
do
for L2M in $(( 2 * L1M ))
do
for L2N in $(( 3 * L1N ))
do
for L2K in $(( 3 * L1K ))
do
L3M=840
L3N=1512
L3K=1344
for i in 0 1 2 3 4 5 6 7 8 9
do
echo "nonsq-$AVX-$L1M-$L1N-$L1K-$L2M-$L2N-$L2K-$L3M-$L3N-$L3K.txt"
make MY_OPT="-O4 -DSIMD_$AVX -DCACHELAYOUT -DL1_BLOCK_SIZE_M=$L1M -DL1_BLOCK_SIZE_N=$L1N -DL1_BLOCK_SIZE_K=$L1K -DL2_BLOCK_SIZE_M=$L2M -DL2_BLOCK_SIZE_N=$L2N -DL2_BLOCK_SIZE_K=$L2K -DL3_BLOCK_SIZE_M=$L3M -DL3_BLOCK_SIZE_N=$L3N -DL3_BLOCK_SIZE_K=$L3K" 1>/dev/null 2>&1
./benchmark-blocked >data/data$i-$AVX-$L1M-$L1N-$L1K-$L2M-$L2N-$L2K-$L3M-$L3N-$L3K.txt
done
done
done
done
done
done
done
