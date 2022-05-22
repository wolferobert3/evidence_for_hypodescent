from os import listdir, system

#Set hyperparameters for producing synthetics
NETWORK_PKL = f'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl'
SAVE_VIDEO = 0
NUM_STEPS = 125

#Set source, output directories
SOURCE = f'/stylegan2-ada-pytorch/cfd/cfd3/divided_images/N_processed/'
targets = listdir(SOURCE)
outdirs = [f'out_source/{target}' for target in targets]

#Simple script for running StyleGAN2-ADA projector file from the command line to produce synthetics for all targets
for idx, target in enumerate(targets):
    system(f'python3 projector.py --save-video 0 --num-steps {NUM_STEPS} --outdir {outdirs[idx]} --target {SOURCE}{target} --network {NETWORK_PKL}')