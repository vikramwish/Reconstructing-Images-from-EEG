

1. create a python environment
We used `requirement.txt`. This was created according to our system specifications.
```
virtualenv pyenv --python=3.10.12
source pyenv/bin/activate
pip install -r requirements.txt
```

2. Get raw data from https://osf.io/3jk45/overview 

Then unzip the files

```
cd data/
wget https://files.de-1.osf.io/v1/resources/anp5v/providers/osfstorage/?zip=
mv index.html?zip= thingseeg2_preproc.zip
unzip thingseeg2_preproc.zip -d thingseeg2_preproc
cd thingseeg2_preproc/
unzip sub-01.zip sub-05.zip sub-08.zip sub-07.zip 

cd ../../
python3 preparation/prepare_thingseeg2_data.py 
```

3. Download the ground truth images - https://osf.io/3jk45/overview

Then unzip "training_images", "test_images" under data/thingseeg2_metadata
```
cd data/
wget https://files.de-1.osf.io/v1/resources/y63gw/providers/osfstorage/?zip=
mv index.html?zip= thingseeg2_metadata.zip
unzip thingseeg2_metadata.zip -d thingseeg2_metadata
cd thingseeg2_metadata/
unzip training_images.zip
unzip test_images.zip
cd ../../
python3 thingseeg2_data_preparation_scripts/save_thingseeg2_images.py
python3 thingseeg2_data_preparation_scripts/save_thingseeg2_concepts.py
```

4. Download  the VDVAE and Versatile Diffusion weights
```
cd vdvae/model/
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-log.jsonl
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model-ema.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-opt.th
cd ../../versatile_diffusion/pretrained/
wget https://huggingface.co/shi-labs/versatile-diffusion/resolve/main/pretrained_pth/vd-four-flow-v1-0-fp16-deprecated.pth
wget https://huggingface.co/shi-labs/versatile-diffusion/resolve/main/pretrained_pth/kl-f8.pth
wget https://huggingface.co/shi-labs/versatile-diffusion/resolve/main/pretrained_pth/optimus-vae.pth
cd ../../
```

5. Extract train and test latent embeddings from images and text labels
```
python3 preparation/vdvae_extract_features.py 
python3 preparation/clipvision_extract_features.py 
python3 preparation/cliptext_extract_features.py 
python3 preparation/evaluation_extract_features_from_test_images.py 
```
6.  For Band Analysis

Run the file with providing the arguments. 
```
python3 preparation/run_bands.py -sfreq 250
```
7. For Electrode analysis. Run separately for each electrode combination and number of channels
```
python3 preparation/run_electrodes.py -sfreq 250
```

8. Training and reconstruction
```
python3 training/train_regression.py 
python3 training/reconstruct_from_embeddings.py 
python3 training/evaluate_reconstruction.py 
python3 training/plot_reconstructions.py -ordered True
python3 training/umap_f.py
```


## Code reused or adapted from the following sources:
1. Code for Step 3, 4, 5,8 were adpated from the repo: https://github.com/desa-lab/Perceptogram.

2. Fei, T., Uppal, A., Jackson, I., Ravishankar, S., Wang, D., & de Sa, V. R. (2024). Perceptogram: Reconstructing Visual Percepts from EEG. arXiv preprint arXiv:2404.01250.

3. Gifford, A. T., Dwivedi, K., Roig, G., & Cichy, R. M. (2022). A large and rich EEG dataset for modeling human visual object recognition. NeuroImage, 264, 119754. https://doi.org/10.1016/j.neuroimage.2022.119754

4. Hebart, M. N., Contier, O., Teichmann, L., Rockter, A. H., Zheng, C. Y., Kidder, A., Corriveau, A., Vaziri-Pashkam, M., & Baker, C. I. (2023). THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior. eLife, 12, e82580. https://doi.org/10.7554/eLife.82580



