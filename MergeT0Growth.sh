#!/bin/sh

# to make file executable : chmod +x MergeT0Growth
# to execute file : ./MergeT0Growth.sh

growth_path=221103_doublechoc/croissance/2xchoc/
t0_path=221103_doublechoc/t0/2xchoc/


tile_nb=18
nb_deb=19


dir_path_growth=/run/media/emuller/Elise_these/Chip/$growth_path
dir_path_t0=/run/media/emuller/Elise_these/Chip/$t0_path

# copy t0 into growth_path


for ((i=$nb_deb; i <$tile_nb+$nb_deb; i++))
do
    cd $dir_path_t0
    file_of_interest=$(find $(cd ..; pwd) -name "*($i)*" -type f)
    echo $file_of_interest
    
    
    cd $dir_path_growth
    directory_of_interest=$(find . -name "*($i)*" -type d)
    mv $file_of_interest $directory_of_interest
    
    cd $directory_of_interest
    t01_file=$(find . -name "Experiment-1020(${i})_EDF_EDF_s0t0c0x0-4248y0-2832.tif" -type f)
    mv $t01_file "./Experiment-1020(${i})_EDF_EDF_s0t01c0x0-4248y0-2832.tif" 
    
    t00_file=$(find . -name "*1018*" -type f)
    mv $t00_file "./Experiment-1020(${i})_EDF_EDF_s0t00c0x0-4248y0-2832.tif" 
    
done

