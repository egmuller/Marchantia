tile_nb=15
nb_deb=19


#dir_path=/run/media/emuller/Elise_these/Chip/$experiment_path
dir_path=/run/media/emuller/Elise_these/Chip/2023/230808_tak1_tak2/Tak2/

# copy files to stitching directory
cd $dir_path

mkdir --verbose -p stitching
files_list=$(find . -name '*t05*' -type f)
cp $files_list stitching
