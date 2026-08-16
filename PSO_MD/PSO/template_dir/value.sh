# !/bin/bash


Bulk_1=$(grep "Bulk Modulus" elastic1/log.lammps | awk '{print $4}' | tail -n 1)
C11_1=$(grep "C11 Modulus" elastic1/log.lammps | awk '{print $4}' | tail -n 1)
C12_1=$(grep "C12 Modulus" elastic1/log.lammps | awk '{print $4}' | tail -n 1)
C44_1=$(grep "C44 Modulus" elastic1/log.lammps | awk '{print $4}' | tail -n 1)
Youngs_1=$(grep "Youngs Modulus" elastic1/log.lammps | awk '{print $4}' | tail -n 1)

Bulk_2=$(grep "Bulk Modulus" elastic2/log.lammps | awk '{print $4}' | tail -n 1)
C11_2=$(grep "C11 Modulus" elastic2/log.lammps | awk '{print $4}' | tail -n 1)
C12_2=$(grep "C12 Modulus" elastic2/log.lammps | awk '{print $4}' | tail -n 1)
C44_2=$(grep "C44 Modulus" elastic2/log.lammps | awk '{print $4}' | tail -n 1)
Youngs_2=$(grep "Youngs Modulus" elastic2/log.lammps | awk '{print $4}' | tail -n 1)

Bulk_3=$(grep "Bulk Modulus" elastic3/log.lammps | awk '{print $4}' | tail -n 1)
C11_3=$(grep "C11 Modulus" elastic3/log.lammps | awk '{print $4}' | tail -n 1)
C12_3=$(grep "C12 Modulus" elastic3/log.lammps | awk '{print $4}' | tail -n 1)
C44_3=$(grep "C44 Modulus" elastic3/log.lammps | awk '{print $4}' | tail -n 1)
Youngs_3=$(grep "Youngs Modulus" elastic3/log.lammps | awk '{print $4}' | tail -n 1)


Energy1_1=$(grep -B 100 '\s\s100\s' GSFE1/log.lammps | awk '{print $4}' | awk 'BEGIN{x=2147483648};$0<x{x=$0};END{print x}')
Energy2_1=$(grep -B 100 '\s\s100\s' GSFE1/log.lammps | awk '{print $4}' | awk 'BEGIN{x=-2147483648};$0>x{x=$0};END{print x}')
E_usf_1=`echo "scale=4; ($Energy2_1 - $Energy1_1)" | bc ` 

Energy1_2=$(grep -B 100 '\s\s100\s' GSFE2/log.lammps | awk '{print $4}' | awk 'BEGIN{x=2147483648};$0<x{x=$0};END{print x}')
Energy2_2=$(grep -B 100 '\s\s100\s' GSFE2/log.lammps | awk '{print $4}' | awk 'BEGIN{x=-2147483648};$0>x{x=$0};END{print x}')
E_usf_2=`echo "scale=4; ($Energy2_2 - $Energy1_2)" | bc ` 

Energy1_3=$(grep -B 100 '\s\s100\s' GSFE3/log.lammps | awk '{print $4}' | awk 'BEGIN{x=2147483648};$0<x{x=$0};END{print x}')
Energy2_3=$(grep -B 100 '\s\s100\s' GSFE3/log.lammps | awk '{print $4}' | awk 'BEGIN{x=-2147483648};$0>x{x=$0};END{print x}')
E_usf_3=`echo "scale=4; ($Energy2_3 - $Energy1_3)" | bc `


Bulk=`echo "scale=4; (($Bulk_1 + $Bulk_2 + $Bulk_3)/3)" | bc `
C11=`echo "scale=4; (($C11_1 + $C11_2 + $C11_3)/3)" | bc `
C12=`echo "scale=4; (($C12_1 + $C12_2 + $C12_3)/3)" | bc `
C44=`echo "scale=4; (($C44_1 + $C44_2 + $C44_3)/3)" | bc `
Youngs=`echo "scale=4; (($Youngs_1 + $Youngs_2 + $Youngs_3)/3)" | bc `
E_usf=`echo "scale=4; (($E_usf_1 + $E_usf_2 + $E_usf_3)/3)" | bc `
 

if [[ $Bulk =~ ^[+-]?[0-9]+\.?[0-9]*$ ]] && [[ $C11 =~ ^[+-]?[0-9]+\.?[0-9]*$ ]] && [[ $C12 =~ ^[+-]?[0-9]+\.?[0-9]*$ ]] && [[ $C44 =~ ^[+-]?[0-9]+\.?[0-9]*$ ]] && [[ $Youngs =~ ^[+-]?[0-9]+\.?[0-9]*$ ]] && [[ $E_usf =~ ^[+-]?[0-9]+\.?[0-9]*$ ]];
then
	echo $Bulk $C11 $C12 $C44 $Youngs $E_usf > result.txt
	echo $Bulk_1 $C11_1 $C12_1 $C44_1 $Youngs_1 >> elastic1.txt
	echo $Bulk_2 $C11_2 $C12_2 $C44_2 $Youngs_2 >> elastic2.txt
	echo $Bulk_3 $C11_3 $C12_3 $C44_3 $Youngs_3 >> elastic3.txt
	echo $E_usf_1 $E_usf_2 $E_usf_3 >> gsfe.txt

else
	echo '1000 1000 1000 1000 1000 1000' > result.txt
	echo '1000 1000 1000 1000 1000' >> elastic1.txt
	echo '1000 1000 1000 1000 1000' >> elastic2.txt
	echo '1000 1000 1000 1000 1000' >> elastic3.txt
	echo '1000 1000 1000' >> gsfe.txt
fi

