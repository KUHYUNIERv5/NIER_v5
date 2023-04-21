# GPU_LIST: 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7

# python train_validation.py --region R4_59 --root_dir /home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU --data_dir /home/pink/dust/NIER_v5-main/data_folder --gpu_list 0 1 2 0 1 2 0 1 2 0 1 2  
# python train_validation.py --region R4_61 --root_dir /home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU --data_dir /home/pink/dust/NIER_v5-main/data_folder --gpu_list 3 4 5 3 4 5 3 4 5 3 4 5

# python train_validation.py --region R4_62 --root_dir /home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU --data_dir /home/pink/dust/NIER_v5-main/data_folder --gpu_list 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7

python inference.py --region='R4_59' --root_dir='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU' --data_dir='/home/pink/dust/NIER_v5-main/data_folder' --device='cuda:0' -i 2022 
python inference.py --region='R4_59' --root_dir='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU' --data_dir='/home/pink/dust/NIER_v5-main/data_folder' --device='cuda:0' -i 2021 

python inference.py --region='R4_60' --root_dir='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU' --data_dir='/home/pink/dust/NIER_v5-main/data_folder' --device='cuda:1' -i 2022 
python inference.py --region='R4_60' --root_dir='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU' --data_dir='/home/pink/dust/NIER_v5-main/data_folder' --device='cuda:0' -i 2021 

python inference.py --region='R4_61' --root_dir='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU' --data_dir='/home/pink/dust/NIER_v5-main/data_folder' --device='cuda:2' -i 2022 
python inference.py --region='R4_61' --root_dir='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU' --data_dir='/home/pink/dust/NIER_v5-main/data_folder' --device='cuda:0' -i 2021 

python inference.py --region='R4_62' --root_dir='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU' --data_dir='/home/pink/dust/NIER_v5-main/data_folder' --device='cuda:3' -i 2022 
python inference.py --region='R4_62' --root_dir='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU' --data_dir='/home/pink/dust/NIER_v5-main/data_folder' --device='cuda:0' -i 2021 

python inference.py --region='R4_63' --root_dir='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU' --data_dir='/home/pink/dust/NIER_v5-main/data_folder' --device='cuda:4' -i 2022 
python inference.py --region='R4_63' --root_dir='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU' --data_dir='/home/pink/dust/NIER_v5-main/data_folder' --device='cuda:0' -i 2021 

python inference.py --region='R4_64' --root_dir='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU' --data_dir='/home/pink/dust/NIER_v5-main/data_folder' --device='cuda:5' -i 2022 
python inference.py --region='R4_64' --root_dir='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU' --data_dir='/home/pink/dust/NIER_v5-main/data_folder' --device='cuda:0' -i 2021 

python inference.py --region='R4_65' --root_dir='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU' --data_dir='/home/pink/dust/NIER_v5-main/data_folder' --device='cuda:6' -i 2022 
python inference.py --region='R4_65' --root_dir='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU' --data_dir='/home/pink/dust/NIER_v5-main/data_folder' --device='cuda:0' -i 2021 

python inference.py --region='R4_66' --root_dir='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU' --data_dir='/home/pink/dust/NIER_v5-main/data_folder' --device='cuda:7' -i 2022 
python inference.py --region='R4_66' --root_dir='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU' --data_dir='/home/pink/dust/NIER_v5-main/data_folder' --device='cuda:0' -i 2021 

python inference.py --region='R4_67' --root_dir='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU' --data_dir='/home/pink/dust/NIER_v5-main/data_folder' --device='cuda:0' -i 2022 
python inference.py --region='R4_67' --root_dir='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU' --data_dir='/home/pink/dust/NIER_v5-main/data_folder' --device='cuda:0' -i 2021 

python inference.py --region='R4_68' --root_dir='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU' --data_dir='/home/pink/dust/NIER_v5-main/data_folder' --device='cuda:0' -i 2022 
python inference.py --region='R4_68' --root_dir='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU' --data_dir='/home/pink/dust/NIER_v5-main/data_folder' --device='cuda:0' -i 2021 
