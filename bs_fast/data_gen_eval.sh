source ~/.bashrc
conda activate e2e
cd ../
python carla_data_gen.py --collection_mode eval --x_range 0 --y_range 0 --x_step 0 --y_step 0