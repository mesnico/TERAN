import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import tqdm
import yaml

from data import get_coco_image_retrieval_data, QueryEncoder
from models.loss import AlignmentContrastiveLoss
from models.teran import TERAN
from utils import AverageMeter, LogCollector


def persist_img_embs(config, data_loader, dataset_indices, numpy_img_emb):
    dst_root = Path(os.getcwd()).joinpath(config['image-retrieval']['pre_computed_img_embeddings_root'])
    if not dst_root.exists():
        dst_root.mkdir(parents=True, exist_ok=True)

    assert len(dataset_indices) == len(numpy_img_emb)
    img_names = get_image_names(dataset_indices, data_loader)
    # TODO do we want to store them in one big npz?
    for idx in range(len(img_names)):
        dst = dst_root.joinpath(img_names[idx] + '.npz')
        if dst.exists():
            continue
        np.savez_compressed(str(dst), img_emb=numpy_img_emb[idx])


def encode_data_for_inference(model: TERAN, data_loader, log_step=10, logging=print, pre_compute_img_embs=False):
    # compute the embedding vectors v_i, s_j (paper) for each image region and word respectively
    # -> forwarding the data through the respective TE stacks
    print(
        f'{"Pre-" if pre_compute_img_embs else ""}Computing image {"" if pre_compute_img_embs else "and query "}embeddings...')

    # we don't need autograd for inference
    model.eval()

    # array to keep all the embeddings
    # TODO maybe we can store those embeddings in an index and load it instead of computing each time for each query
    query_embs = None
    num_query_feats = None
    num_img_feats = None  # all images have a fixed size of pre-extracted features of 36 + 1 regions
    img_embs = None

    # make sure val logger is used
    batch_time = AverageMeter()
    val_logger = LogCollector()
    model.logger = val_logger

    start_time = time.time()
    for i, (img_feature_batch, img_feat_bboxes_batch, img_feat_len_batch, query_token_batch, query_len_batch,
            dataset_indices) in enumerate(data_loader):
        batch_start_time = time.time()
        """
        the data loader returns None values for the respective batches if the only query was already loaded 
        -> query_token_batch, query_len_batch = None, None
        """

        with torch.no_grad():
            # compute the query embedding only in the first iteration (also because there is only 1 query in IR)
            if query_embs is None and not pre_compute_img_embs:
                # TODO maybe we can get the most matching roi from query_emb_aggr?
                query_emb_aggr, query_emb, _ = model.forward_txt(query_token_batch, query_len_batch)

                # store results as np arrays for further processing or persisting
                num_query_feats = query_len_batch[0] if isinstance(query_len_batch, list) else query_len_batch
                query_feat_dim = query_emb.size(2)
                query_embs = torch.zeros((1, num_query_feats, query_feat_dim))
                query_embs[0, :, :] = query_emb.cpu().permute(1, 0, 2)

            # compute every image embedding in the dataset
            img_emb_aggr, img_emb = model.forward_img(img_feature_batch, img_feat_len_batch, img_feat_bboxes_batch)

            # init array to store results for further processing or persisting
            if img_embs is None:
                num_img_feats = img_feat_len_batch[0] if isinstance(img_feat_len_batch,
                                                                    list) else img_feat_len_batch
                img_feat_dim = img_emb.size(2)
                img_embs = torch.zeros((len(data_loader.dataset), num_img_feats, img_feat_dim))

            numpy_img_emb = img_emb.cpu().permute(1, 0, 2)  # why are we permuting here? -> TERAN
            img_embs[dataset_indices, :, :] = numpy_img_emb
            if pre_compute_img_embs:
                # if we are in a pre-compute run, persist the arrays
                persist_img_embs(model_config, data_loader, dataset_indices, numpy_img_emb)

        # measure elapsed time per batch
        batch_time.update(time.time() - batch_start_time)

        if i % log_step == 0:
            logging(
                f"Batch: [{i}/{len(data_loader)}]\t{str(model.logger)}\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})")
        del img_feature_batch, query_token_batch

    print(
        f"Time elapsed to {'encode' if not pre_compute_img_embs else 'encode and persist'} data: {time.time() - start_time} seconds.")
    return img_embs, query_embs, num_img_feats, num_query_feats


def compute_distances(img_embs, query_embs, img_lengths, query_lengths, config):
    # initialize similarity matrix evaluator
    sim_matrix_fn = AlignmentContrastiveLoss(aggregation=config['image-retrieval']['alignment_mode'],
                                             return_similarity_mat=True)
    start_time = time.time()
    img_emb_batches = 1  # TODO config / calc
    img_embs_per_batch = img_embs.size(0) // img_emb_batches  # TODO config variable

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
        img_embs_batch = img_embs[i * img_embs_per_batch:(i + 1) * img_embs_per_batch]
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


def get_image_names(dataset_indices, dataset) -> List[str]:
    return [dataset.get_image_metadata(idx)[1]['file_name'] for idx in dataset_indices]


def load_precomputed_image_embeddings(config):
    print("Loading pre-computed image embeddings...")
    start = time.time()
    # returns a PreComputedCocoImageEmbeddingsDataset
    dataset = get_coco_image_retrieval_data(config)

    # get the img embeddings and convert them to Tensors
    np_img_embs = np.array(list(dataset.img_embs.values()))
    img_embs = torch.Tensor(np_img_embs)  # here is the bottleneck
    img_lengths = len(np_img_embs[0])
    print(f"Time elapsed to load pre-computed embeddings and compute query embedding: {time.time() - start} seconds!")
    return img_embs, img_lengths, dataset


def top_k_image_retrieval(opts, config, checkpoint) -> List[str]:
    # construct model
    model = TERAN(config)

    # load model state
    model.load_state_dict(checkpoint['model'], strict=False)

    use_precomputed_img_embeddings = config['image-retrieval']['use_precomputed_img_embeddings']
    if use_precomputed_img_embeddings:
        # load pre computed img embs
        img_embs, img_lengths, dataset = load_precomputed_image_embeddings(config)
        # compute query emb
        query_encoder = QueryEncoder(config, model)
        query_embs, query_lengths = query_encoder.compute_query_embedding(opts.query)

    else:
        # returns a Dataloader of a PreComputedCocoFeaturesDataset
        data_loader = get_coco_image_retrieval_data(config,
                                                    query=opts.query,
                                                    num_workers=opts.num_data_workers)
        dataset = data_loader.dataset
        # encode the data (i.e. compute the embeddings / TE outputs for the images and query)
        img_embs, query_embs, img_lengths, query_lengths = encode_data_for_inference(model, data_loader)

    if opts.device == "cuda":
        torch.cuda.empty_cache()

    print(f"Images Embeddings: {img_embs.shape[0]}, Query Embeddings: {query_embs.shape[0]}")

    # compute the matching scores
    distance_sorted_indices = compute_distances(img_embs, query_embs, img_lengths, query_lengths, config)
    top_k_indices = distance_sorted_indices[:opts.top_k]

    # get the image names
    top_k_images = get_image_names(top_k_indices, dataset)
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


def pre_compute_img_embeddings(opts, config, checkpoint):
    # construct model
    model = TERAN(config)

    # load model state
    model.load_state_dict(checkpoint['model'], strict=False)

    print('Loading dataset')
    data_loader = get_coco_image_retrieval_data(config,
                                                query=opts.query,
                                                num_workers=opts.num_data_workers,
                                                pre_compute_img_embs=True)

    # encode the data (i.e. compute the embeddings / TE outputs for the images and query)
    encode_data_for_inference(model, data_loader, pre_compute_img_embs=True)


if __name__ == '__main__':
    print("CUDA_VISIBLE_DEVICES: " + os.getenv("CUDA_VISIBLE_DEVICES", "NOT SET - ABORTING"))
    if os.getenv("CUDA_VISIBLE_DEVICES", None) is None:
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help="Model (checkpoint) to load. E.g. pretrained_models/coco_MrSw.pth.tar", required=True)
    parser.add_argument('--pre_compute_img_embeddings', action='store_true', help="If set or true, the image "
                                                                                  "embeddings get precomputed and "
                                                                                  "persisted at the directory "
                                                                                  "specified in the config.")
    parser.add_argument('--query', type=str, required='--pre_compute_img_embeddings' not in sys.argv)
    parser.add_argument('--num_data_workers', type=int, default=8)
    parser.add_argument('--num_images', type=int, default=5000)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--dataset', type=str, choices=['coco'], default='coco')  # TODO support other datasets
    parser.add_argument('--config', type=str, default='configs/teran_coco_MrSw_IR.yaml', help="Which configuration to "
                                                                                              "use for overriding the"
                                                                                              " checkpoint "
                                                                                              "configuration. See "
                                                                                              "into 'config' folder")
    # cpu is only for local test runs
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    opts = parser.parse_args()

    model_config, model_checkpoint = prepare_model_checkpoint_and_config(opts)

    if not opts.pre_compute_img_embeddings:
        top_k_matches = top_k_image_retrieval(opts, model_config, model_checkpoint)
        print(f"##########################################")
        print(f"QUERY: {opts.query}")
        print(f"######## TOP {opts.top_k} RESULTS ########")
        print(top_k_matches)
    else:
        pre_compute_img_embeddings(opts, model_config, model_checkpoint)
