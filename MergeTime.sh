#!/bin/sh


#t0_path=221128_choc_osmotique/choc-t15min/aller/500mM/
#t10_path=221128_choc_osmotique/choc-t15min/retour/500mM/
#t20_path=221118_choc_osmotique/choc_25uLpromin_10minDelay/t20-25/

#destination_path=221128_choc_osmotique/choc-t15min/All/500mM/

tile_nb=36
nb_deb=1

dir_path_t0=/run/media/emuller/TRANSCEND/221128_choc_osmotique/t0/
#dir_path_t10=/run/media/emuller/TRANSCEND/221128_choc_osmotique/croissance_t23h30-t30h/
#dir_path_t20=/run/media/emuller/Elise_these/Chip/$t20_path

dir_path_destination=/run/media/emuller/TRANSCEND/221128_choc_osmotique/croissance/

#cd $dir_path_t0
#file_of_interest=$(find . -name "*ORG*" -type f)
#rm $file_of_interest

#cd $dir_path_t10
#file_of_interest=$(find . -name "*ORG*" -type f)
#rm $file_of_interest

#cd $dir_path_t20
#file_of_interest=$(find . -name "*ORG*" -type f)
#rm $file_of_interest

#cd $dir_path_destination
#file_of_interest=$(find . -name "*ORG*" -type f)
#rm $file_of_interest

#cp -RT $dir_path_t0 $dir_path_destination
#cp -r $dir_path_t0 $dir_path_destination

for ((i=$nb_deb; i <$tile_nb+$nb_deb; i++))
do
    cd $dir_path_t0
    file_of_interest=$(find $(cd ..; pwd) -name "*1068($i)*" -type f)
    #echo $file_of_interest
    
    #cd $dir_path_t20
    #file_of_interest2=$(find $(cd ..; pwd) -name "*($i)*" -type f)
    #echo $file_of_interest2
    
    cd $dir_path_destination
    directory_of_interest=$(find . -name "*($i)*" -type d)
    cp $file_of_interest $directory_of_interest
    #mv $file_of_interest2 $directory_of_interest
    
    
done







