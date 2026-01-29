
1. Create a Python Conda (3. 9.2) environment. 
Use the environment.yaml for this. We created it according to our system specifications.

2. Download data from https://huggingface.co/datasets/Alljoined/Alljoined-1.6M and https://openneuro.org/datasets/ds005106/versions/1.5.0

- Unzip the files. 
- Run the file
```
CUDA_VISBILE_DEVICES=0,1,2,3 python3 preprocessing.py --subjects 1-20

```
3. For Phase 1,
```
CUDA_VISBILE_DEVICES=0,1,2,3 python3 stage1_harmonise.py --datasets [dataset path] --output_dir [output path] num_subjects 20 
```
4. For Phase 2
```
CUDA_VISBILE_DEVICES=0,1,2,3 python3 stage2_eeg_vit.py --output_dir [output path]
```
5. For Phase 3 - Get access to DINOv3 model from - https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m and download it 
```
python3 stage3_alignment.py
```
6. Extract the embeddings from trained aligned model 
```
python3 extract_stage3.py               
```

7. For Phase 4 - Reconstruction 
```
CUDA_VISIBLE_DEVICES=4 python3 stage4_reconstruct.py --image_dir [your image_path] --output_path [result path]
```


# Citation

1. Xu, J., Nunes, U. B., Jiang, W., Ryther, S., Pringle, J., Scotti, P. S., ... & Kneeland, R. (2025). Alljoined-1.6 M: A Million-Trial EEG-Image Dataset for Evaluating Affordable Brain-Computer Interfaces.

2. Tijl Grootswagers, Genevieve Quek, Zhen Zeng, Manuel Varlet. 2025. “Human Infant EEG Recordings for 200 Object Images Presented in Rapid Visual Streams.” Scientific Data.

3. https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m




