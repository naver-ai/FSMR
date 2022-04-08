# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import argparse
import os
import pickle
import re

import numpy as np
import PIL.Image

import dnnlib
import dnnlib.tflib as tflib
import tensorflow as tf

from tqdm import tqdm
#----------------------------------------------------------------------------
"""
    dataset = 'cifar10'
    model = ['stylegan2', 'stylegan2_fsmr', 'ada', 'ada_fsmr', 'diffaug', 'diffaug_fsmr']
    session = ['123', '118', '244', '117', '132', '133']
    snap = ['014131', '044032', '065536', '052224', '096051', '065945']
    
    dataset = 'ffhq'
    model = ['stylegan2', 'stylegan2_fsmr', 'ada', 'ada_fsmr', 'diffaug', 'diffaug_fsmr']
    session = ['125', '113', '94', '114', '130', '132']
    snap = ['014541', '024781', '014555', '024985', '025000', '024166']
    
    dataset = 'afhq_gan'
    model = ['stylegan2', 'stylegan2_fsmr', 'ada', 'ada_fsmr', 'diffaug', 'diffaug_fsmr']
    session = ['27', '29', '24', '25', '36', '38']
    snap = ['025000', '007987', '005749', '023347', '025000', '013721']
    
    dataset = 'afhq_dog'
    model = ['stylegan2', 'stylegan2_fsmr', 'ada', 'ada_fsmr', 'diffaug', 'diffaug_fsmr']
    session = ['11', '12', '13', '14', '19', '20']
    snap = ['001638', '001638', '025000', '015360', '009625', '004096']
    
    dataset = 'afhq_cat'
    model = ['stylegan2', 'stylegan2_fsmr', 'ada', 'ada_fsmr', 'diffaug', 'diffaug_fsmr']
    session = ['12', '13', '14', '18', '19', '20']
    snap = ['002457', '001228', '025000', '020684', '024780', '008396']
    
    dataset = 'metface'
    model = ['stylegan2', 'stylegan2_fsmr', 'ada', 'ada_fsmr', 'diffaug', 'diffaug_fsmr']
    session = ['29', '28', '30', '31', '32', '35']
    snap = ['000409', '000614', '020275', '020070', '001638', '002048']
     
"""
def generate_images(network_pkl, seeds, truncation_psi, outdir, class_idx, dlatents_npz):

    def save_image_grid(images, filename, drange, grid_size):
        images = images.transpose(0, 3, 1, 2)
        lo, hi = drange
        gw, gh = grid_size
        images = np.asarray(images, dtype=np.float32)
        images = (images - lo) * (255 / (hi - lo))
        images = np.rint(images).clip(0, 255).astype(np.uint8)
        print(images.shape)
        _N, C, H, W = images.shape
        images = images.reshape(gh, gw, C, H, W)
        images = images.transpose(0, 3, 1, 4, 2)
        images = images.reshape(gh * H, gw * W, C)
        PIL.Image.fromarray(images, {3: 'RGB', 1: 'L'}[C]).save(filename)

    truncation_psi = None
    # seeds = [7, 77, 777, 7777, 77777, 777777]
    seeds = [s for s in range(5000)]
    dataset = 'cifar10'
    model = ['stylegan2', 'stylegan2_fsmr', 'ada', 'ada_fsmr', 'diffaug', 'diffaug_fsmr']
    session = ['123', '118', '244', '117', '132', '133']
    snap = ['014131', '044032', '065536', '052224', '096051', '065945']
    # network_pkl = 'checkpoint/stylegan2_{}/{}_{}_{}/network-snapshot-{}.pkl'.format(dataset, model, dataset, session, snap)

    tflib.init_tf()

    network_pkl_list = []

    for i in range(len(model)):
        network_pkl_list += ['checkpoint/stylegan2_{}/{}_{}_{}/network-snapshot-{}.pkl'.format(dataset, model[i], dataset, session[i], snap[i])]

    for idx, network_pkl in enumerate(network_pkl_list):
        print('Loading networks from "%s"...' % network_pkl)
        with dnnlib.util.open_url(network_pkl) as fp:
            _G, _D, Gs = pickle.load(fp)

        os.makedirs(outdir, exist_ok=True)

        # Render images for a given dlatent vector.
        if dlatents_npz is not None:
            print(f'Generating images from dlatents file "{dlatents_npz}"')
            dlatents = np.load(dlatents_npz)['dlatents']
            assert dlatents.shape[1:] == (18, 512) # [N, 18, 512]
            imgs = Gs.components.synthesis.run(dlatents, output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))
            for i, img in enumerate(imgs):
                fname = f'{outdir}/dlatent{i:02d}.png'
                print (f'Saved {fname}')
                PIL.Image.fromarray(img, 'RGB').save(fname)
            return

        # Render images for dlatents initialized from random seeds.
        Gs_kwargs = {
            'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
            'randomize_noise': False
        }
        if truncation_psi is not None:
            Gs_kwargs['truncation_psi'] = truncation_psi

        if not dataset == 'cifar10':
            noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
            label = np.zeros([1] + Gs.input_shapes[1][1:])
            if class_idx is not None:
                label[:, class_idx] = 1

            for seed in tqdm(seeds):
                # print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
                rnd = np.random.RandomState(seed)
                z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
                tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
                images = Gs.run(z, label, **Gs_kwargs) # [minibatch, height, width, channel]
                # PIL.Image.fromarray(images[0], 'RGB').save('{}/fid_results/{}/{}/{}_{}_{}_{}.png'.format(outdir, dataset, model[idx], dataset, model[idx], seed, truncation_psi))
                PIL.Image.fromarray(images[0], 'RGB').save('C:/Users/takis/dataset/real_fake/{}_fake/{}/{}_{}_{}_{}.png'.format(dataset, model[idx], dataset, model[idx], seed, truncation_psi))
        else:
            (gw, gh) = (32, 32)
            grid_size = (gw, gh)

            noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
            label = np.zeros([gw * gh] + Gs.input_shapes[1][1:])
            if class_idx is not None:
                label[:, class_idx] = 1

            # rnd = np.random.RandomState(777)
            # z = rnd.randn(gw * gh, *Gs.input_shape[1:])  # [minibatch, component]
            z = np.random.randn(np.prod(grid_size), *Gs.input_shape[1:])
            tflib.set_vars({var: np.random.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
            images = Gs.run(z, label, **Gs_kwargs)  # [minibatch, height, width, channel]

            f_name = 'out/sample_results/{}/{}/{}_{}_{}.png'.format(dataset, model[idx], dataset, model[idx], truncation_psi)
            save_image_grid(images, filename=f_name, drange=[0, 255], grid_size=grid_size)

def uncurate_figures(network_pkl, seeds, truncation_psi, outdir, class_idx, dlatents_npz):
    dataset = 'ffhq_20%'
    model = 'diffaug_fsmr'
    session = '21'
    snap = '011264'
    network_pkl = 'checkpoint/stylegan2_{}/{}_{}_{}/network-snapshot-{}.pkl'.format(dataset, model, dataset, session, snap)

    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    os.makedirs(outdir, exist_ok=True)

    # Render images for dlatents initialized from random seeds.
    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
    }
    if truncation_psi is not None:
        Gs_kwargs['truncation_psi'] = truncation_psi

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    label = np.zeros([1] + Gs.input_shapes[1][1:])
    if class_idx is not None:
        label[:, class_idx] = 1

    z_dim = 512
    img_size = 256
    seed = 777

    lods = [0, 1, 2, 2, 3, 3]
    rows = 3
    cx = 0
    cy = 0

    latents = np.random.RandomState(seed).normal(size=[sum(rows * 2 ** lod for lod in lods), z_dim])

    tflib.set_vars({var: np.random.RandomState(seed).randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]

    images = Gs.run(latents, label, **Gs_kwargs)

    # for i in range(len(images)):
    #     images[i] = post_process_generator_output(images[i])

    canvas = PIL.Image.new('RGB', (sum(img_size // 2 ** lod for lod in lods), img_size * rows), 'white')
    image_iter = iter(list(images))

    for col, lod in enumerate(lods):
        for row in range(rows * 2 ** lod):
            image = PIL.Image.fromarray(np.uint8(next(image_iter)), 'RGB')

            image = image.crop((cx, cy, cx + img_size, cy + img_size))
            image = image.resize((img_size // 2 ** lod, img_size // 2 ** lod), PIL.Image.ANTIALIAS)
            canvas.paste(image, (sum(img_size // 2 ** lod for lod in lods[:col]), row * img_size // 2 ** lod))

    canvas.save('{}/uncurate_results/{}/uncurated_{}_{}_{}_{}.jpg'.format(outdir, dataset, dataset, model, seed, truncation_psi))

#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

_examples = '''examples:

  # Generate curated MetFaces images without truncation (Fig.10 left)
  python %(prog)s --outdir=out --trunc=1 --seeds=85,265,297,849 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

  # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
  python %(prog)s --outdir=out --trunc=0.7 --seeds=600-605 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

  # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
  python %(prog)s --outdir=out --trunc=1 --seeds=0-35 --class=1 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/cifar10.pkl

  # Render image from projected latent vector
  python %(prog)s --outdir=out --dlatents=out/dlatents.npz \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate images using pretrained network pickle.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--network', help='Network pickle filename', dest='network_pkl')
    # g = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('--seeds', type=_parse_num_range, help='List of random seeds')
    parser.add_argument('--dlatents', dest='dlatents_npz', help='Generate images for saved dlatents')
    parser.add_argument('--trunc', dest='truncation_psi', type=float, help='Truncation psi (default: %(default)s)', default=0.7)
    parser.add_argument('--class', dest='class_idx', type=int, help='Class label (default: unconditional)')
    parser.add_argument('--outdir', help='Where to save the output images', metavar='DIR', default='out')

    args = parser.parse_args()
    generate_images(**vars(args))
    # uncurate_figures(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
