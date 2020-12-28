import argparse
import os
import sys

import torch
import yaml

import evaluation


def main(opt, current_config):
    model_checkpoint = opt.checkpoint

    if opt.gpu:
        checkpoint = torch.load(model_checkpoint)  # , map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(model_checkpoint, map_location=torch.device("cpu"))

    print('Checkpoint loaded from {}'.format(model_checkpoint))
    loaded_config = checkpoint['config']

    if opt.size == "1k":
        fold5 = True
    elif opt.size == "5k":
        fold5 = False
    else:
        raise ValueError('Test split size not recognized!')

    # Override some mandatory things in the configuration (paths)
    if current_config is not None:
        loaded_config['dataset']['images-path'] = current_config['dataset']['images-path']
        loaded_config['dataset']['data'] = current_config['dataset']['data']
        loaded_config['image-model']['pre-extracted-features-root'] = current_config['image-model'][
            'pre-extracted-features-root']
        loaded_config['training']['bs'] = current_config['training']['bs']

    evaluation.evalrank(loaded_config, checkpoint, split="test", fold5=False, eval_t2i=opt.t2i, eval_i2t=opt.i2t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help="Checkpoint to load")
    parser.add_argument('--size', type=str, choices=['1k', '5k'], default='1k')
    parser.add_argument('--gpu', type=bool, default=True, help="If false, CPU is used for computations; GPU otherwise.")
    parser.add_argument('--t2i', action='store_true', default=True,
                        help="If set text-to-image (image retrieval) evaluation will be executed.")
    parser.add_argument('--i2t', action='store_true', default=False,
                        help="If set image-to-text (image captioning) evaluation will be executed.")
    parser.add_argument('--config', type=str, default=None, help="Which configuration to use for overriding the "
                                                                 "checkpoint configuration. See into 'config' folder")

    print("CUDA_VISIBLE_DEVICES: " + os.getenv("CUDA_VISIBLE_DEVICES", "NOT SET - ABORTING"))
    if os.getenv("CUDA_VISIBLE_DEVICES", None) is None:
        sys.exit(1)

    opt = parser.parse_args()
    if opt.config is not None:
        with open(opt.config, 'r') as ymlfile:
            config = yaml.load(ymlfile)
    else:
        config = None
    main(opt, config)
