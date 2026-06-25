#!/bin/bash

arg=$1
epoch=$2
cd $arg

cp data100_use1.dat ./file_save/data_100_use1_$epoch.dat
cp data100_use2.dat ./file_save/data_100_use2_$epoch.dat
cp data100_use3.dat ./file_save/data_100_use3_$epoch.dat

cp data111_use1.dat ./file_save111/data_111_use1_$epoch.dat
cp data111_use2.dat ./file_save111/data_111_use2_$epoch.dat
cp data111_use3.dat ./file_save111/data_111_use3_$epoch.dat

cd ../

