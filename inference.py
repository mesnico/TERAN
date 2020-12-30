import argparse
import os
import sys
import time
from typing import List, Any, Dict

import numpy as np
import torch
import tqdm
import yaml

from data import get_coco_image_retrieval_data_loader
from models.loss import AlignmentContrastiveLoss
from models.teran import TERAN
from utils import AverageMeter, LogCollector


def encode_data_for_inference(model: TERAN, data_loader, log_step=10, logging=print):
    # compute the embedding vectors v_i, s_j (paper) for each image region and word respectively
    # -> forwarding the data through the respective TE stacks
    print('Computing image and query embeddings...')
    encode_data_start_time = time.time()

    batch_time = AverageMeter()
    val_logger = LogCollector()

    # we don't need autograd for inference
    model.eval()

    # array to keep all the embeddings
    # TODO maybe we can store those embeddings in an index and load it instead of computing each time for each query
    query_embs = None
    num_query_feats = None
    num_img_feats = None  # all images have a fixed size of pre-extracted features of 36 + 1 regions
    img_embs = None

    start_time = time.time()
    for i, (img_feature_batch, img_feat_bboxes_batch, img_feat_lengths, query_token_ids, query_lengths,
            dataset_indices) in enumerate(data_loader):

        # make sure val logger is used
        model.logger = val_logger

        # TODO
        # in the first version just stack the query_token_ids, img_feat_length and query_length
        # so that it has shape B x ? x ?, where B is len(img_feature_batch) (should be equal to bs set in the config)
        #
        # in the second version adapt model.forward_emb so that the embeddings get only computed once and then stacked
        # to the same size as the img_embs

        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        with torch.no_grad():
            # TODO inside model.forward_emb we have to adapt the code for only a single query so that it doesn't get
            # computed each time
            _, _, img_emb, query_emb, _ = model.forward_emb(img_feature_batch,
                                                            query_token_ids,
                                                            img_feat_lengths,
                                                            query_lengths,
                                                            img_feat_bboxes_batch)

            # initialize the arrays given the size of the embeddings
            if img_embs is None:
                num_img_feats = img_feat_lengths[0] if isinstance(img_feat_lengths, list) else img_feat_lengths
                num_query_feats = query_lengths[0] if isinstance(query_lengths, list) else query_lengths
                img_feat_dim = img_emb.size(2)
                query_feat_dim = query_emb.size(2)
                img_embs = torch.zeros((len(data_loader.dataset), num_img_feats, img_feat_dim))
                query_embs = torch.zeros((len(data_loader.dataset), num_query_feats, query_feat_dim))

            # preserve the embeddings by copying from gpu and converting to numpy
            img_embs[dataset_indices, :, :] = img_emb.cpu().permute(1, 0, 2)
            query_embs[dataset_indices, :, :] = query_emb.cpu().permute(1, 0, 2)

        # measure elapsed time per batch
        batch_time.update(time.time() - start_time)
        start_time = time.time()

        if i % log_step == 0:
            logging(
                f"Batch: [{i}/{len(data_loader)}]\t{str(model.logger)}\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})")
        del img_feature_batch, query_token_ids

    print(f"Time elapsed to encode data: {time.time() - encode_data_start_time} seconds.")
    return img_embs, query_embs, num_img_feats, num_query_feats


def compute_distance_sorted_indices(img_embs, query_embs, img_lengths, query_lengths, config):
    # initialize similarity matrix evaluator
    sim_matrix_fn = AlignmentContrastiveLoss(aggregation=config['image-retrieval']['alignment_mode'],
                                             return_similarity_mat=True)
    start_time = time.time()
    img_embs_per_batch = 1000  # TODO config variable
    img_emb_batches = 5  # TODO config / calc

    num_img_embs = img_embs.shape[0]

    # distances storage
    distances = None

    # since its always the same query we can reuse the batch
    # (TODO maybe we can even just use a batch of size 1?! -> check the sim_matrix_fn)
    query_emb_batch = query_embs[:1]
    query_length_batch = [query_lengths[0] if isinstance(query_lengths, list) else query_lengths for _ in range(1)]
    query_emb_batch.cuda()

    # batch-wise compute the alignment distance between the images and the query
    for i in tqdm.trange(img_emb_batches):
        # create the current batch
        img_embs_batch = img_embs[i * img_embs_per_batch:(i+1) * img_embs_per_batch]
        img_embs_length_batch = [img_lengths for _ in range(img_embs_per_batch)]
        img_embs_batch.cuda()

        # compute and pool the similarity matrices to get the global distance between the image and the query
        alignment_distance = sim_matrix_fn(img_embs_batch, query_emb_batch, img_embs_length_batch, query_length_batch)
        alignment_distance = alignment_distance.t().cpu().numpy()

        # store the distances
        if distances is None:
            distances = alignment_distance
        else:
            distances = np.concatenate([distances, alignment_distance], axis=1)

    # get the img indices descended sorted by the distance matrix
    sorted_distance_indices = np.argsort(distances.squeeze())[::-1]
    print(f"Time elapsed to compute and pool the similarity matrices: {time.time() - start_time} seconds.")
    return sorted_distance_indices


def get_image_names(top_k_indices, data_loader) -> List[str]:
    return [data_loader.dataset.get_image_metadata(idx)['file_name'] for idx in top_k_indices]


def top_k_image_retrieval(opts, config, checkpoint) -> List[str]:
    # load model and options
    # checkpoint = torch.load(model_path)
    data_path = config['dataset']['data']
    measure = config['training']['measure']

    # construct model
    model = TERAN(config)

    # load model state
    model.load_state_dict(checkpoint['model'], strict=False)

    print('Loading dataset')
    data_loader = get_coco_image_retrieval_data_loader(config,
                                                       query=opts.query,
                                                       workers=opts.num_data_workers)

    # encode the data (i.e. compute the embeddings / TE outputs for the images and query)
    img_embs, cap_embs, img_lengths, cap_lengths = encode_data_for_inference(model, data_loader)

    torch.cuda.empty_cache()
    print(f"Images: {img_embs.shape[0]}, Captions: {cap_embs.shape[0]}")

    # compute the matching scores
    distance_sorted_indices = compute_distance_sorted_indices(img_embs, cap_embs, img_lengths, cap_lengths, config)
    top_k_indices = distance_sorted_indices[:opts.top_k]

    # get the image names
    top_k_images = get_image_names(top_k_indices, data_loader)
    return top_k_images


def prepare_model_checkpoint_and_config(opts):
    checkpoint = torch.load(opts.model, map_location=torch.device(opts.device))
    print('Checkpoint loaded from {}'.format(opts.model))
    model_checkpoint_config = checkpoint['config']

    with open(opts.config, 'r') as yml_file:
        loaded_config = yaml.load(yml_file)
        # Override some mandatory things in the configuration
        model_checkpoint_config['dataset']['images-path'] = loaded_config['dataset']['images-path']
        model_checkpoint_config['dataset']['data'] = loaded_config['dataset']['data']
        model_checkpoint_config['image-retrieval'] = loaded_config['image-retrieval']

    return model_checkpoint_config, checkpoint


if __name__ == '__main__':
    print("CUDA_VISIBLE_DEVICES: " + os.getenv("CUDA_VISIBLE_DEVICES", "NOT SET - ABORTING"))
    if os.getenv("CUDA_VISIBLE_DEVICES", None) is None:
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help="Model (checkpoint) to load. E.g. pretrained_models/coco_MrSw.pth.tar", required=True)
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')  # cpu is only for local test runs
    parser.add_argument('--num_data_workers', type=int, default=8)
    parser.add_argument('--num_images', type=int, default=5000)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--dataset', type=str, choices=['coco'], default='coco')  # TODO support other datasets
    parser.add_argument('--config', type=str, default='configs/teran_coco_MrSw_IR.yaml',
                        help="Which configuration to use for overriding the checkpoint configuration. See into "
                             "'config' folder")
    opts = parser.parse_args()

    model_config, model_checkpoint = prepare_model_checkpoint_and_config(opts)

    top_k_matches = top_k_image_retrieval(opts, model_config, model_checkpoint)

    print(f"######## TOP {opts.top_k} RESULTS ########")
    print(top_k_matches)
