#!/bin/bash

 arg=$1
 cd $arg

  w1=$(cat new_par.txt | awk '{ print $1}')
  w2=$(cat new_par.txt | awk '{ print $2}')  
  w3=$(cat new_par.txt | awk '{ print $3}')
  w4=$(cat new_par.txt | awk '{ print $4}')
  w5=$(cat new_par.txt | awk '{ print $4}')

 # we need to set the five values 
  sed -e "s/\${i}/1/" -e "s/\${w1}/$w1/" template.prm >  tt1
  sed -e "s/\${i}/1/" -e "s/\${w2}/$w2/" tt1 >  tt2  
  sed -e "s/\${i}/1/" -e "s/\${w3}/$w3/" tt2 >  tt3
  sed -e "s/\${i}/1/" -e "s/\${w4}/$w4/" tt3 >  tt4  
  sed -e "s/\${i}/1/" -e "s/\${w5}/$w5/" tt4 >  structure.gen.sh
  rm tt1 tt2 tt3 tt4
  bash structure.gen.sh
cd ../

