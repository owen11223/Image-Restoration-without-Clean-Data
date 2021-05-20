# CS577 Project
# Noise2Noise - Image Restoration without Clean Data
# Ji Tin Justin Li - A20423037
# Subowen Yan - A20430537

# Run instructions to replicate the experiments

# 1. clone the repository
# 2. navigate to the current project directory
# 3. Download the image to the local machine by running

~/.../prj $ python src/download_kodak --output-dir imgs/kodak
~/.../prj $ python src/download_BSD300.py

# Since there was some package dependency issues with PIL on the xsede container img,
# the text written noisy data is recommended be prepared on the local machine with:

~/.../prj $ python src/data.py

# to run the job with sbatch on xsede, login to the xsede login node, 
and scp the entire project directory to the login node.
~/.../ $ ssh [USERNAME]@comet.sdsc.xsede.org

# on the cluster, scp with
~/ $ scp -r [localUsername]@[localIp]:~/.../prj ~/.

# to train a single model, allocate a container
srun --partition=gpu-shared --gres=gpu:1 --pty --nodes=1 --ntasks-per-node=1 -t 01:00:00 --wait=0 --export=ALL /bin/bash
# and run with
singularity exec --nv /share/apps/gpu/singularity/images/keras/keras-v2.2.4-tensorflow-v1.12-gpu-20190214.simg python3 src/train_model.py g kodak 
# which would train a model that restores gaussian noise images, using the kodak image set

# available arguments for train_model.py are:
# python3 train_modle.py [noisetype] [imgset]
# [noisetype]:
#   g - gaussian noise
#   p - poisson noise
#   b - multiplicative bernoulli noise
#   t - text overlay noise
# [imgset]:
#   kodak - for 24 imgs kodak set
#   bsds300 - for 200 imgs bsds300 set

# To train all the models as a batch, 
# navigate to this dir, and run
~/prj $ sbatch prj_test.sb
# which would run and train all 8 models.

# This creates the loss plots in the "plots" directory.
# And the trained models are saved in the "models" directory.

# To do inference with the trained models, 

# python inference.py

# It assumes the test image in /BSDS300/images/test






