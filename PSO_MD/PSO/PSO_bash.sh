
#! /bin/bash

   arg=$1
 
  nbirds=$(echo $arg-1 | bc -l)
  

  rm -f launch_job.txt
  for i in $(seq 0 $nbirds)
  do 
   echo 'cd '$i/elastic1' ; hostname > tt; module load <fftw_module>; <lammps_binary> -in elastic.in' >> launch_job.txt
  done;


  for i in $(seq 0 $nbirds)
  do 
   echo 'cd '$i/elastic2' ; hostname > tt; module load <fftw_module>; <lammps_binary> -in elastic.in' >> launch_job.txt
  done;
  
 
  for i in $(seq 0 $nbirds)
  do 
   echo 'cd '$i/elastic3' ; hostname > tt; module load <fftw_module>; <lammps_binary> -in elastic.in' >> launch_job.txt
  done;

  for i in $(seq 0 $nbirds)
  do 
   echo 'cd '$i/GSFE1' ; hostname > tt; module load <fftw_module>; <lammps_binary> -in GSFE.in' >>  launch_job.txt
  done;

  for i in $(seq 0 $nbirds)
  do 
   echo 'cd '$i/GSFE2' ; hostname > tt; module load <fftw_module>; <lammps_binary> -in GSFE.in' >>  launch_job.txt
  done;
  
   for i in $(seq 0 $nbirds)
  do 
   echo 'cd '$i/GSFE3' ; hostname > tt; module load <fftw_module>; <lammps_binary> -in GSFE.in' >>  launch_job.txt
  done;


 parallel --jobs 8 --workdir $PWD --sshloginfile nodelist.txt <  launch_job.txt

 wait


 rm -f result.txt
 for i in $(seq 0 $nbirds)
  do
   echo 'cd '$i' ; sh value.sh' >> result.txt
  done;

 parallel --jobs 1 --workdir $PWD --sshloginfile nodelist.txt < result.txt
 
wait 
