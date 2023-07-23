#!/bin/sh


#t0_path=221128_choc_osmotique/choc-t15min/aller/500mM/
#t10_path=221128_choc_osmotique/choc-t15min/retour/500mM/
#t20_path=221118_choc_osmotique/choc_25uLpromin_10minDelay/t20-25/

#destination_path=221128_choc_osmotique/choc-t15min/All/500mM/

tile_nb=54
nb_deb=1
#nb_deb_t5=19


dir_path_t5=/run/media/emuller/TRANSCEND/Data/Chip/230524_storedgrowth_EF1aPME113.1/block3/
#dir_path_t3=/run/media/emuller/TRANSCEND/Data/Chip/230321_mri_tak_chocs/croissance/tak/
#dir_path_t4=/run/media/emuller/TRANSCEND/Data/Chip/230224_PI0_takmri/mriGb/t7h/
#dir_path_t5=/run/media/emuller/Elise_4T/Data/Chip/2023/230321_mri_tak_chocs/croissance/tak/

dir_path_destination=/run/media/emuller/TRANSCEND/Data/Chip/230524_storedgrowth_EF1aPME113.1/block1/


#cp -RT $dir_path_t0 $dir_path_destination
#cp -r $dir_path_t0 $dir_path_destination

for ((i=$nb_deb; i <$tile_nb+$nb_deb; i++))
do
    
    #cd $dir_path_t2
    #file_of_interest2=$(find $(cd ..; pwd) -name "*1230($i)*t0*" -type f)
    
    #cd $dir_path_t3
    #file_of_interest3=$(find $(cd ..; pwd) -name "*1228($i)*" -type f)
    
    #cd $dir_path_t4
    #file_of_interest4=$(find $(cd ..; pwd) -name "*1184($i)*" -type f)
    
    cd $dir_path_t5
    #ibis=$(($nb_deb_t5+$i-1))
    echo $i
    file_of_interest5=$(find $(cd ..; pwd) -name "*Block3($i)*" -type f)
    
    cd $dir_path_destination
    directory_of_interest=$(find . -name "*($i)*" -type d)
    #cp $file_of_interest2 $directory_of_interest
    #cp $file_of_interest3 $directory_of_interest
    #cp $file_of_interest4 $directory_of_interest
    cp $file_of_interest5 $directory_of_interest
    
done







