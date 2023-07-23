#!/bin/sh

# to make file executable : chmod +x Stitching.sh
# to execute file : ./Stitching.sh

#experiment_path=221128_croissance_PME/PME/
tile_nb=18
nb_deb=1


#dir_path=/run/media/emuller/Elise_these/Chip/$experiment_path
dir_path=/run/media/emuller/Elise_these/Chip/2023/230602_Pi0_tak_mri/mri/choc8h/

# copy files to stitching directory
cd $dir_path

mkdir --verbose -p stitching
files_list=$(find . -name '*t005*' -type f)
cp $files_list stitching

# rename files
cd stitching
for ((i=$nb_deb; i <=$tile_nb+$nb_deb; i++))
do
	echo $i
	res=$(($i%$tile_nb))
	if [[ $res -eq 0 ]];then res=$tile_nb; fi ;
	if [[ $res -lt 10 ]]; then res="0${res}"; fi;
	file_of_interest=$(find . -name "*($i)*" -type f)
	echo $file_of_interest
	mv $file_of_interest "./tile_${res}.tif" 

done


