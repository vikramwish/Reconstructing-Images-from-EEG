
## Getting started
1. create a python environment
In this experiment, we used `requirement.txt`. This was created according to our system specifications

linux:
```
virtualenv pyenv --python=3.10.12
source pyenv/bin/activate
pip install -r requirements.txt
```

2. Download raw data (https://osf.io/anp5v/), unzip "sub01", "sub02"

```
cd data/
wget https://files.de-1.osf.io/v1/resources/anp5v/providers/osfstorage/?zip=
mv index.html?zip= thingseeg2_preproc.zip
unzip thingseeg2_preproc.zip -d thingseeg2_preproc
cd thingseeg2_preproc/
unzip sub-01.zip
unzip sub-02.zip
unzip sub-03.zip
unzip sub-04.zip
unzip sub-05.zip
unzip sub-06.zip
unzip sub-07.zip
unzip sub-08.zip
unzip sub-09.zip
unzip sub-10.zip
cd ../../
python3 preparation/prepare_thingseeg2_data.py 
```

3. Download [ground truth images](https://osf.io/y63gw/), unzip "training_images", "test_images" under data/thingseeg2_metadata
```
cd data/
wget https://files.de-1.osf.io/v1/resources/y63gw/providers/osfstorage/?zip=
mv index.html?zip= thingseeg2_metadata.zip
unzip thingseeg2_metadata.zip -d thingseeg2_metadata
cd thingseeg2_metadata/
unzip training_images.zip
unzip test_images.zip
cd ../../
python thingseeg2_data_preparation_scripts/save_thingseeg2_images.py
python thingseeg2_data_preparation_scripts/save_thingseeg2_concepts.py
```

4. Download VDVAE and Versatile Diffusion weights
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
6. Band Analysis
```
python3 preparation/run_bands.py -sfreq 250
```
7. Electrode combinations
```
python3 preparation/run_electrodes.py -sfreq 250
```

8. Training and reconstruction
```
python3 training/train_regression.py 
python3 training/reconstruct_from_embeddings.py 
python3 training/evaluate_reconstruction.py 
python3 training/plot_reconstructions.py -ordered True
python3 training/plot_umap_CLIP.py
python3 training/umap_f.py
```


# Citations
Ozcelik, F., & VanRullen, R. (2023). Natural scene reconstruction from fMRI signals using generative latent diffusion. Scientific Reports, 13(1), 15666. https://doi.org/10.1038/s41598-023-42891-8

Gifford, A. T., Dwivedi, K., Roig, G., & Cichy, R. M. (2022). A large and rich EEG dataset for modeling human visual object recognition. NeuroImage, 264, 119754. https://doi.org/10.1016/j.neuroimage.2022.119754

Benchetrit, Y., Banville, H., & King, J.-R. (n.d.). BRAIN DECODING: TOWARD REAL-TIME RECONSTRUCTION OF VISUAL PERCEPTION.

Hebart, M. N., Contier, O., Teichmann, L., Rockter, A. H., Zheng, C. Y., Kidder, A., Corriveau, A., Vaziri-Pashkam, M., & Baker, C. I. (2023). THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior. eLife, 12, e82580. https://doi.org/10.7554/eLife.82580



