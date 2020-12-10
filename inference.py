import argparse
from typing import List
from data import get_inference_loader
import torch
import yaml

from models.teran import TERAN


def image_retrieval(checkpoint, opts, config) -> List[str]:
    # load model and options
    # checkpoint = torch.load(model_path)
    data_path = config['dataset']['data']
    measure = config['training']['measure']

    # construct model
    model = TERAN(config)

    # load model state
    model.load_state_dict(checkpoint['model'], strict=False)

    print('Loading dataset')
    dataloader = get_inference_loader(config, opts, workers=4)

    return ["1", "2"]


def main(opts, current_config) -> List[str]:
    checkpoint = torch.load(opts.checkpoint, map_location=torch.device(opts.device))

    print('Checkpoint loaded from {}'.format(opts.checkpoint))
    loaded_config = checkpoint['config']

    # Override some mandatory things in the configuration (paths)
    if current_config is not None:
        loaded_config['dataset']['images-path'] = current_config['dataset']['images-path']
        loaded_config['dataset']['data'] = current_config['dataset']['data']
        loaded_config['image-model']['pre-extracted-features-root'] = current_config['image-model'][
            'pre-extracted-features-root']

    top_k_results = image_retrieval(checkpoint, opts, loaded_config)
    return top_k_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Model (checkpoint) to load. E.g. pretrained_models/coco_MrSw.pth.tar"
                        , required=True)
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--num_images', type=int, default=1000)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--dataset', type=str, choices=['coco', 'flickr30k'], default='coco')
    parser.add_argument('--config', type=str, default=None, help="Which configuration to use for overriding the "
                                                                 "checkpoint configuration. See into 'config' folder")

    opts = parser.parse_args()
    if opts.config is not None:
        with open(opts.config, 'r') as yml_file:
            config = yaml.load(yml_file)
    else:
        config = None
    top_k_results = main(opts, config)
    print(f"######## TOP {opts.tok_k} RESULTS ########")
    print(top_k_results)
