#!/usr/bin/python

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
from pathlib import Path
import argparse
import scipy.spatial

def make_args():
    description = 'get proverbs from gutenberg books'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-f',
                        '--folder',
                        help='path to folder',
                        required=True,
                        type=valid_path)

    parser.add_argument('-i',
                        '--inputfile',
                        help='path to input',
                        required=True,
                        type=valid_path)


    return parser.parse_args()


def valid_path(p):
    return Path(p)


def main(folder, infile):
    # Initialize TensorFlow.
    seed =1000
    #set random seed for tf
    tflib.init_tf({'rnd.np_random_seed': seed})

    # Load pre-trained network.
    with open(str(folder)+'/'+str(infile), 'rb') as file:
        _G, _D, Gs = pickle.load(file)
            # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
            # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
            # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
    
    Gs.print_layers()
        
    # Pick latent vector.
    grid_size = (7, 4)
    #generate random vectors
    grid_latents = [np.random.randn(1, Gs.input_shape[1]) for i in range(np.prod(grid_size))]
    

    # Generate images
    os.mkdir('grid-images/{}'.format(str(infile)[:-4]))
    count=0
    #same args as training_loop.py
    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
    }


    #generate and save images
    for vector in grid_latents:
       # fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(vector, None, **Gs_kwargs)
    
        # Save image.
        png_filename = 'grid-images/{}/{}.png'.format(str(infile)[:-4],count)
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
        count+=1



if __name__ == "__main__":
    args = make_args()
    print('args made')
    folder = args.folder
    infile = args.inputfile
    main(folder,infile)
