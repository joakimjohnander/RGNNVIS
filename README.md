# RGNNVIS
Welcome to the public repository for our work "Learning Video Instance Segmentation with Recurrent Graph Neural Networks". The paper was published at GCPR2021 and a preliminary version is available at https://arxiv.org/pdf/2012.03911 .

# Running the Code
We provide pre-trained weights for the backbone and detector (https://liuonline-my.sharepoint.com/:f:/g/personal/joajo88_ad_liu_se/EpVUTs9wrF1HuNiZxGyRb_kBH6lRv2YEk-7r5KFdrSFb0w?e=iY8ntF) . To run:
```
# Get weights
cp MY_DOWNLOADED_WEIGHTS/* pytorch_weights/.

# Download datasets.
...

# Build singularity container.
cd singularity
docker build -f Dockerfile21.09 -t pytorch21.09:211224 .
singularity build pytorch21_09.sif docker-daemon://pytorch21.09:211224
cd ..

# Setup paths
emacs singularity/rgnnvis.sh  # Configure the paths

# Run training and evaluation
sbatch /workspaces/$USER/RGNNVIS/singularity/rgnnvis.sh w49/rgnnvis_resnet50.py --train --ytvis_test
```
