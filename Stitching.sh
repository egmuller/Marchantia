#!/bin/sh

# to make file executable : chmod +x Stitching.sh
# to execute file : ./Stitching.sh

#experiment_path=221128_croissance_PME/PME/
tile_nb=15
nb_deb=31


#dir_path=/run/media/emuller/Elise_these/Chip/$experiment_path
dir_path=/run/media/emuller/Elise_these/Chip/2023/230617_croissance_EF1_fer5/fer5/

# copy files to stitching directory
cd $dir_path

mkdir --verbose -p stitching
files_list=$(find . -name '*t05*' -type f)
cp $files_list stitching

# rename files
cd stitching
for ((i=$nb_deb; i <=$tile_nb+$nb_deb; i++))
do
	echo $i
	res=$((($i+1-$nb_deb)%$tile_nb))
	if [[ $res -eq 0 ]];then res=$tile_nb; fi ;
	if [[ $res -lt 10 ]]; then res="0${res}"; fi;
	file_of_interest=$(find . -name "*($i)*" -type f)
	echo $file_of_interest
	mv $file_of_interest "./tile_${res}.tif" 

done


