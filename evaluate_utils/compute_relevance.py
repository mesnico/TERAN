from torchvision.datasets.coco import CocoCaptions
import numpy as np
import os
import yaml
import tqdm
import argparse
import multiprocessing
from functools import partial
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
from evaluate_utils import rouge, spice
import utils
import data
import copy


def compute_relevances_wrt_query(query):
    i, (_, query_caption, _, _) = query
    row_dataloader = DataLoader(compute_relevances_wrt_query.dataset,
                                num_workers=0,
                                batch_size=5,
                                shuffle=False,
                                collate_fn=my_collate
                                )
    if compute_relevances_wrt_query.method == 'rougeL':
        # scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scorer = rouge.Rouge()

        for j, (_, cur_captions, _, _) in enumerate(row_dataloader):
            if compute_relevances_wrt_query.npy_file[i, j] < 0:
                # fine-grain check on negative values. If not negative, this value has already been computed
                relevance = scorer.score(query_caption, cur_captions)
                compute_relevances_wrt_query.npy_file[i, j] = relevance

    elif compute_relevances_wrt_query.method == 'spice':
        if any(compute_relevances_wrt_query.npy_file[i, :] < 0):
            scorer = spice.Spice()
            # accumulate all captions
            all_captions = []
            for j, (_, cur_captions, _, _) in enumerate(row_dataloader):
                all_captions.append(cur_captions)

            _, scores = scorer.compute_score(all_captions, query_caption)
            relevances = [s['All']['f'] for s in scores]
            relevances = np.array(relevances)
            compute_relevances_wrt_query.npy_file[i, :] = relevances


def parallel_worker_init(npy_file, dataset, method):
    compute_relevances_wrt_query.npy_file = npy_file
    compute_relevances_wrt_query.dataset = dataset
    compute_relevances_wrt_query.method = method


def get_dataset(config, split):
    roots, ids = data.get_paths(config)

    data_name = config['dataset']['name']
    if 'coco' in data_name:
        # COCO custom dataset
        dataset = data.CocoDataset(root=roots[split]['img'], json=roots[split]['cap'], ids=ids[split], get_images=False)
    elif 'f8k' in data_name or 'f30k' in data_name:
        dataset = data.FlickrDataset(root=roots[split]['img'], split=split, json=roots[split]['cap'], get_images=False)
    return dataset


def my_collate(batch):
    transposed_batch = list(zip(*batch))
    return transposed_batch


def main(args, config):
    dataset = get_dataset(config, args.split)
    queries_dataloader = DataLoader(dataset, num_workers=0,
                                batch_size=1,
                                shuffle=False,
                                collate_fn=my_collate
                                )

    relevance_dir = os.path.join(config['dataset']['data'], config['dataset']['name'], 'relevances')
    if not os.path.exists(relevance_dir):
        os.makedirs(relevance_dir)
    relevance_filename = os.path.join(relevance_dir, '{}-{}-{}.npy'.format(config['dataset']['name'], args.split, args.method))
    if os.path.isfile(relevance_filename):
        answ = input("Relevances for {} already existing in {}. Continue? (y/n)".format(args.method, relevance_filename))
        if answ != 'y':
            quit()

    # filename = os.path.join(cache_dir,'d_{}.npy'.format(query_img_index))
    n_queries = len(queries_dataloader)
    n_images = len(queries_dataloader) // 5
    if os.path.isfile(relevance_filename):
        # print('Graph distances file existing for image {}, cache {}! Loading...'.format(query_img_index, cache_name))
        print('Loading existing file {} with shape {} x {}'.format(relevance_filename, n_queries, n_images))
        npy_file = np.memmap(relevance_filename, dtype=np.float32, shape=(n_queries, n_images), mode='r+')
    else:
        print('Creating new file {} with shape {} x {}'.format(relevance_filename, n_queries, n_images))
        npy_file = np.memmap(relevance_filename, dtype=np.float32, shape=(n_queries, n_images), mode='w+')
        npy_file[:, :] = -1

    # print('Computing {} distances for image {}, cache {}...'.format(n,query_img_index,cache_name))

    # pbar = ProgressBar(widgets=[Percentage(), Bar(), AdaptiveETA()], maxval=n).start()
    print('Starting relevance computation...')
    with multiprocessing.Pool(processes=args.ncpus, initializer=parallel_worker_init,
                              initargs=(npy_file, dataset, args.method)) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(compute_relevances_wrt_query, enumerate(queries_dataloader)), total=n_queries):
            pass


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Extract captioning scores for use as relevance')
    arg_parser.add_argument('--config', type=str, help="Which configuration to use. See into 'config' folder")
    arg_parser.add_argument('--method', type=str, default="rougeL", help="Scoring method")
    arg_parser.add_argument('--split', type=str, default="val", help="Dataset split to use")
    arg_parser.add_argument('--ncpus', type=int, default=12, help="How many gpus to use")

    args = arg_parser.parse_args()
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)
    main(args, config)
