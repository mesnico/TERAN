# This script extract features and put them in shelve format

import os
import torch
import tqdm
import argparse
import yaml
import re
import itertools
import pickle
import numpy as np
from torch.utils.data import DataLoader
# from graphrcnn.extract_features import extract_visual_features
# from torchvision.datasets.coco import CocoCaptions
# from datasets import CocoCaptionsOnly
from torchvision import transforms
from torchvision.models import resnet18, resnet50, resnet101, resnet152, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from transformers import BertTokenizer, BertModel
# from datasets import TextCollator
import shelve
import data
from models.text import EncoderTextBERT


class TextCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched captions.
    This should be passed to the DataLoader
    """

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        # images = transposed_batch[0]
        captions = transposed_batch[1]
        return captions


class FeatureExtractor(object):
    def __init__(self, config, split, bs=1, collate_fn=torch.utils.data.dataloader.default_collate):
        self.config = config
        self.split = split
        self.output_feat_fld = os.path.join(config['dataset']['data'], '{}_precomp'.format(config['dataset']['name']))
        if not os.path.exists(self.output_feat_fld):
            os.makedirs(self.output_feat_fld)

    def extract(self):
        """
        Extracts features and dump them on a db file.
        For text extractors: each file record is a dictionary with keys:
        'image_id' (int) and 'features' (np.array K x dim)
        For image extractors: each file record is a dictionary with keys:
        'boxes' (np.array K x 4), 'scores' (np.array K x 1), 'features' (np.array K x dim)
        :return: void
        """
        raise NotImplementedError

    def get_db_file(self):
        """
        :return: the path to the db file for these features
        """
        raise NotImplementedError


class HuggingFaceTransformerExtractor(FeatureExtractor):
    def __init__(self, config, split, model_name='bert', pretrained='bert-base-uncased', finetuned=None):
        super(HuggingFaceTransformerExtractor, self).__init__(config, split, bs=5, collate_fn=TextCollator())
        self.pretrained = pretrained
        self.finetuned = finetuned
        self.model_name = model_name
        self.config = config

        roots, ids = data.get_paths(config)

        data_name = config['dataset']['name']
        transform = data.get_transform(data_name, 'val', config)
        collate_fn = data.Collate(config)
        self.loader = data.get_loader_single(data_name, split,
                                             roots[split]['img'],
                                             roots[split]['cap'],
                                             transform, ids=ids[split],
                                             batch_size=32, shuffle=False,
                                             num_workers=4, collate_fn=collate_fn)

    def get_db_file(self):
        finetuned_str = "" if not self.finetuned else '_finetuned'
        feat_dst_filename = os.path.join(self.output_feat_fld,
                                         '{}_{}_{}{}.db'.format(self.split, self.model_name, self.pretrained, finetuned_str))
        print('Hugging Face BERT features filename: {}'.format(feat_dst_filename))
        return feat_dst_filename

    def extract(self, device='cuda'):
        # Load pre-trained model tokenizer (vocabulary) and model itself
        if self.model_name == 'bert':
            self.config['text-model']['layers'] = 0
            self.config['text-model']['pre-extracted'] = False
            model = EncoderTextBERT(self.config)
        else:
            raise ValueError('{} model is not known'.format(self.model))

        if self.finetuned:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            checkpoint = torch.load(self.finetuned, map_location=device)['model']
            checkpoint = {k[k.find('.txt_enc.'):].replace('.txt_enc.', ''): v for k, v in checkpoint.items() if '.txt_enc.' in k}
            model.load_state_dict(checkpoint, strict=False)
            print('BERT model extracted from trained model at {}'.format(self.finetuned))

        model.to(device)
        model.eval()

        feat_dst_filename = self.get_db_file()
        prog_id = 0
        with shelve.open(feat_dst_filename, flag='n') as db:
            for images, captions, img_lengths, cap_lengths, boxes, ids in tqdm.tqdm(self.loader):
                captions = captions.cuda()

                with torch.no_grad():
                    _, feats = model(captions, cap_lengths)
                    # get the features from the last hidden state
                    feats = feats.cpu().numpy()
                    word_embs = model.word_embeddings(captions)
                    word_embs = word_embs.cpu().numpy()
                    for c, f, w, l, i in zip(captions.cpu().numpy(), feats, word_embs, cap_lengths, ids):
                        # dump_feats.append(f[:l])
                        dump_dict = {'image_id': i, 'captions': c, 'features': f[:l], 'wembeddings': w[:l]}
                        db[str(prog_id)] = dump_dict
                        prog_id += 1


class TextWordIndexesExtractor(FeatureExtractor):
    def __init__(self, dataset, root, split):
        super().__init__(dataset, root, split)

    def get_db_file(self):
        feat_dst_filename = os.path.join(self.output_feat_fld,
                                         '{}_{}_word_indexes.db'.format(self.dataset, self.split))
        return feat_dst_filename

    def extract(self, device='cuda'):
        if self.dataset == 'coco':
            dataset_root = os.path.join(self.root, '{}2014'.format(self.split))
            dataset_json = os.path.join(self.root,
                                            'stanford_split_annots', 'captions_{}2014.json'.format(self.split))
            dataset = CocoCaptionsOnly(dataset_root, dataset_json, indexing='images')

            dataloader = DataLoader(dataset,
                                    num_workers=4,
                                    batch_size=1,
                                    shuffle=False,
                                    )
        else:
            raise ValueError('{} dataset is not implemented!'.format(self.dataset))

        # Build dictionary
        dict_file_path = os.path.join(self.output_feat_fld, 'word_dict_{}.pkl'.format(self.dataset))
        if not os.path.isfile(dict_file_path):
            if not self.split == 'train':
                raise ValueError('Dictionary should be built on the train set. Rerun with split=train')
            else:
                print('Building dictionary ...')
                wdict = {}
                wfreq = {}
                counter = 2  # 1 is the unknown label
                for i, captions in enumerate(tqdm.tqdm(dataloader)):
                    captions = [c[0] for c in captions]
                    tokenized_captions = [re.sub('[!#?,.:";]', '', c).strip().replace("'", ' ').lower().split(' ') for c in captions]
                    words = itertools.chain.from_iterable(tokenized_captions)
                    for w in words:
                        # create dict
                        if w not in wdict:
                            wdict[w] = counter
                            counter += 1

                        # handle word frequencies
                        if w not in wfreq:
                            wfreq[w] = 1
                        else:
                            wfreq[w] += 1

                print('Filtering dictionary ...')
                # Filter dictionary based on frequencies
                for w, f in wfreq.items():
                    if f == 1:
                        wdict[w] = 1  # 1 is the unknown label

                with open(dict_file_path, 'wb') as f:
                    pickle.dump(wdict, f)
        else:
            print('Loading dict from {}'.format(dict_file_path))
            with open(dict_file_path, 'rb') as f:
                wdict = pickle.load(f)

        feat_dst_filename = self.get_db_file()
        prog_id = 0
        with shelve.open(feat_dst_filename, flag='n') as db:
            for i, captions in enumerate(tqdm.tqdm(dataloader)):
                captions = [c[0] for c in captions]
                tokenized_captions = [re.sub('[!#?,.:";]', '', c).strip().replace("'", ' ').lower().split(' ') for c in captions]
                lengths = [len(c) for c in tokenized_captions]
                max_len = max(lengths)

                for tc, l in zip(tokenized_captions, lengths):
                    indexes = [wdict[w] if w in wdict else 1 for w in tc]
                    # dump_feats.append(f[:l])
                    dump_dict = {'image_id': i, 'features': np.expand_dims(np.asarray(indexes), axis=1)}
                    db[str(prog_id)] = dump_dict
                    prog_id += 1


class ResnetFeatureExtractor(FeatureExtractor):
    def __init__(self, dataset, root, split, resnet_depth, output_dims=(1, 1)):
        super().__init__(dataset, root, split)
        self.resnet_depth = resnet_depth
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_dims)
        self.output_dims = output_dims

    def extract(self, device='cuda'):
        if self.dataset == 'coco':
            dataset_root = os.path.join(self.root, '{}2014'.format(self.split))
            dataset_json = os.path.join(self.root,
                                        'stanford_split_annots', 'captions_{}2014.json'.format(self.split))
            if self.split == 'train':
                transform = transforms.Compose(
                    [transforms.Resize(256),
                     transforms.FiveCrop(224),
                     transforms.Lambda(lambda crops: torch.stack([
                         transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ])(crop) for crop in crops])
                     )])
            elif self.split == 'test' or self.split == 'val':
                transform = transforms.Compose(
                    [transforms.Resize(256),
                     transforms.CenterCrop(224),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]),
                     transforms.Lambda(lambda imgt: imgt.unsqueeze(0))
                     ]
                )

            dataset = CocoCaptions(dataset_root, dataset_json,
                                   transform=transform)
            dataloader = DataLoader(dataset,
                                    num_workers=4,
                                    batch_size=1,
                                    shuffle=False
                                    )
        if self.resnet_depth == 18:
            model = resnet18(pretrained=True)
        elif self.resnet_depth == 50:
            model = resnet50(pretrained=True)
        elif self.resnet_depth == 101:
            model = resnet101(pretrained=True)
        elif self.resnet_depth == 152:
            model = resnet152(pretrained=True)

        # delete the classification and the pooling layers
        modules = list(model.children())[:-2]
        model = torch.nn.Sequential(*modules)
        model.to(device)
        model.eval()

        feat_dst_filename = self.get_db_file()
        with shelve.open(feat_dst_filename, flag='n') as db:
            for idx, (img, _) in enumerate(tqdm.tqdm(dataloader)):
                with torch.no_grad():
                    img = img.to(device)
                    img = img.squeeze(0)
                    feats = model(img)
                    feats = self.avgpool(feats)
                    feats = feats.view(feats.shape[0], feats.shape[1], -1)
                    feats = feats.permute(0, 2, 1).squeeze(0)
                    if idx == 0:
                        print('Features have shape {}'.format(feats.shape))
                    dump_dict = {'scores': None, 'boxes': None, 'features': feats.cpu().numpy()}
                    db[str(idx)] = dump_dict

    def get_db_file(self):
        feat_dst_filename = os.path.join(self.output_feat_fld,
                                         '{}_{}_resnet{}_{}x{}.db'.format(self.dataset, self.split,
                                                                          self.resnet_depth, self.output_dims[0],
                                                                          self.output_dims[1]))
        return feat_dst_filename


class VGGFeatureExtractor(FeatureExtractor):
    def __init__(self, dataset, root, split, vgg_depth):  # , output_dims=(1, 1)):
        super().__init__(dataset, root, split)
        self.vgg_depth = vgg_depth
        # self.avgpool = torch.nn.AdaptiveAvgPool2d(output_dims)
        # self.output_dims = output_dims

    def extract(self, device='cuda'):
        if self.dataset == 'coco':
            dataset_root = os.path.join(self.root, '{}2014'.format(self.split))
            dataset_json = os.path.join(self.root,
                                        'stanford_split_annots', 'captions_{}2014.json'.format(self.split))
            if self.split == 'train':
                transform = transforms.Compose(
                    [transforms.Resize(256),
                     transforms.FiveCrop(224),
                     transforms.Lambda(lambda crops: torch.stack([
                         transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ])(crop) for crop in crops])
                     )])
            elif self.split == 'test' or self.split == 'val':
                transform = transforms.Compose(
                    [transforms.Resize(256),
                     transforms.CenterCrop(224),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]),
                     transforms.Lambda(lambda imgt: imgt.unsqueeze(0))
                     ]
                )

            dataset = CocoCaptions(dataset_root, dataset_json,
                                   transform=transform)
            dataloader = DataLoader(dataset,
                                    num_workers=4,
                                    batch_size=1,
                                    shuffle=False
                                    )
        if self.vgg_depth == 11:
            model = vgg11_bn(pretrained=True)
        elif self.vgg_depth == 13:
            model = vgg13_bn(pretrained=True)
        elif self.vgg_depth == 16:
            model = vgg16_bn(pretrained=True)
        elif self.vgg_depth == 19:
            model = vgg19_bn(pretrained=True)

        # delete the classification and the pooling layers
        modules = list(model.classifier.children())[:-3]
        model.classifier = torch.nn.Sequential(*modules)
        model.to(device)
        model.eval()

        feat_dst_filename = self.get_db_file()
        with shelve.open(feat_dst_filename, flag='n') as db:
            for idx, (img, _) in enumerate(tqdm.tqdm(dataloader)):
                with torch.no_grad():
                    img = img.to(device)
                    img = img.squeeze(0)
                    feats = model(img)
                    # feats = self.avgpool(feats)
                    # feats = feats.view(feats.shape[0], feats.shape[1], -1)
                    # feats = feats.permute(0, 2, 1).squeeze(0)
                    if self.split == 'train':
                        feats = feats.unsqueeze(1)
                    if idx == 0:
                        print('Features have shape {}'.format(feats.shape))
                    dump_dict = {'scores': None, 'boxes': None, 'features': feats.cpu().numpy()}
                    db[str(idx)] = dump_dict

    def get_db_file(self):
        feat_dst_filename = os.path.join(self.output_feat_fld,
                                         '{}_{}_vgg{}_bn.db'.format(self.dataset, self.split,
                                                                          self.vgg_depth))
        return feat_dst_filename


# class GraphRcnnFeatureExtractor(FeatureExtractor):
#     def __init__(self, dataset, root, split, algorithm='sg_imp'):
#         super().__init__(dataset, root, split)
#         self.algorithm = algorithm
#
#     def extract(self):
#         # use the graphrcnn package to extract visual relational features
#         extract_visual_features(self.dataset, self.root, self.algorithm, self.split)
#
#     def get_db_file(self):
#         feat_dst_filename = os.path.join(self.output_feat_fld,
#                                          '{}_{}_{}.db'.format(self.dataset, self.split,
#                                                               self.algorithm))
#         return feat_dst_filename


def get_features_extractor(config, split, method=None, finetuned=None):
    if method == 'transformer-bert':
        config['text-model']['pre-extracted'] = False
        extractor = HuggingFaceTransformerExtractor(config, split, finetuned=finetuned)

    # elif method == 'graphrcnn':
    #     extractor = GraphRcnnFeatureExtractor(dataset_name, dataset_root, split,
    #                                           extractor_config['algorithm'])
    # elif method == 'resnet':
    #     extractor = ResnetFeatureExtractor(dataset_name, dataset_root, split,
    #                                        extractor_config['depth'], (extractor_config['output-h'],
    #                                                                    extractor_config['output-w']))
    # elif method == 'vgg':
    #     extractor = VGGFeatureExtractor(dataset_name, dataset_root, split,
    #                                     extractor_config['depth'])
    else:
        raise ValueError('Extraction method {} not known!'.format(args.method))
    return extractor


def main(args, config):
    extractor = get_features_extractor(config, args.split, args.method, args.finetuned)
    if os.path.isfile(extractor.get_db_file() + '.dat'):
        answ = input("Features {} for {} already existing. Overwrite? (y/n)".format(extractor.get_db_file(), extractor))
        if answ == 'y':
            print('Using extractor: {}'.format(extractor))
            extractor.extract()
        else:
            print('Skipping {}'.format(extractor))
    else:
        extractor.extract()

    print('DONE')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Extract captioning scores for use as relevance')
    arg_parser.add_argument('--config', type=str, help="Which configuration to use. See into 'config' folder")
    arg_parser.add_argument('--split', type=str, default="val", help="Dataset split to use")
    arg_parser.add_argument('--finetuned', type=str, default=None, help="Optional finetuning checkpoint")
    arg_parser.add_argument('method', type=str, help="Which kind of feature you want to extract")
    # arg_parser.add_argument('type', type=str, choices=['image','text'], help="Method type")

    args = arg_parser.parse_args()

    if args.finetuned is not None:
        config = torch.load(args.finetuned)['config']
        print('Configuration read from checkpoint')
    else:
        with open(args.config, 'r') as ymlfile:
            config = yaml.load(ymlfile)
    main(args, config)

