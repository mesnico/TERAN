import json as jsonmod
import os
import pickle

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import tqdm
from PIL import Image
from pycocotools.coco import COCO
from transformers import BertTokenizer

from features import HuggingFaceTransformerExtractor


def get_paths(config):
    # noinspection PyIncorrectDocstring
    # noinspection PyUnresolvedReferences
    """
        Returns paths to images and annotations for the given datasets. For MSCOCO
        indices are also returned to control the data split being used.
        The indices are extracted from the Karpathy et al. splits using this
        snippet:

        >>> import json
        >>> dataset=json.load(open('dataset_coco.json','r'))
        >>> A=[]
        >>> for i in range(len(D['images'])):
        ...   if D['images'][i]['split'] == 'val':
        ...     A+=D['images'][i]['sentids'][:5]
        ...

        :param name: Dataset names
        :param use_restval: If True, the the `restval` data is included in train.
        """
    name = config['dataset']['name']
    annotations_path = os.path.join(config['dataset']['data'], name, 'annotations')
    use_restval = config['dataset']['restval']

    roots = {}
    ids = {}
    if 'coco' == name:
        imgdir = config['dataset']['images-path']
        capdir = annotations_path
        roots['train'] = {
            'img': os.path.join(imgdir, 'train2014'),
            'cap': os.path.join(capdir, 'captions_train2014.json')
        }
        roots['val'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }
        ids['train'] = np.load(os.path.join(annotations_path, 'coco_train_ids.npy'))
        ids['val'] = np.load(os.path.join(annotations_path, 'coco_dev_ids.npy'))[:5000]
        ids['test'] = np.load(os.path.join(annotations_path, 'coco_test_ids.npy'))
        ids['trainrestval'] = (
            ids['train'],
            np.load(os.path.join(annotations_path, 'coco_restval_ids.npy'))
        )
        if use_restval:
            roots['train'] = roots['trainrestval']
            ids['train'] = ids['trainrestval']
    elif 'f30k' == name:
        imgdir = config['dataset']['images-path']
        cap = os.path.join(annotations_path, 'dataset_flickr30k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}

    return roots, ids


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, imgs_root, captions_json, transform=None, coco_annotation_ids=None, get_images=True):
        """
        Args:
            imgs_root: image directory.
            captions_json: coco annotation file path.
            transform: transformer for image.
        """
        self.root = imgs_root
        self.get_images = get_images
        # when using `restval`, two json files are needed
        if isinstance(captions_json, tuple):
            self.coco = (COCO(captions_json[0]), COCO(captions_json[1]))
        else:
            self.coco = (COCO(captions_json),)
            self.root = (imgs_root,)
        # if ids provided by get_paths, use split-specific ids
        if coco_annotation_ids is None:
            self.annotation_ids = list(self.coco[0].anns.keys())
        else:
            self.annotation_ids = coco_annotation_ids

        # if `restval` data is to be used, record the break point for ids
        if isinstance(self.annotation_ids, tuple):
            self.bp = len(self.annotation_ids[0])
            self.annotation_ids = list(self.annotation_ids[0]) + list(self.annotation_ids[1])
        else:
            self.bp = len(self.annotation_ids)
        self.transform = transform

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        root, caption, img_id, path, image, _ = self.get_raw_item(index, self.get_images)

        if self.transform is not None:
            image = self.transform(image)

        target = caption
        return image, target, index, img_id

    def get_raw_item(self, index, load_image=True):
        if index < self.bp:  # bp -> breakpoint to stop after N samples
            coco = self.coco[0]
            root = self.root[0]
        else:
            coco = self.coco[1]
            root = self.root[1]
        ann_id = self.annotation_ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        img_metadata = coco.imgs[img_id]
        img_size = np.array([img_metadata['width'], img_metadata['height']])
        if load_image:
            path = coco.loadImgs(img_id)[0]['file_name']
            image = Image.open(os.path.join(root, path)).convert('RGB')

            return root, caption, img_id, path, image, img_size
        else:
            return root, caption, img_id, None, None, img_size

    def __len__(self):
        return len(self.annotation_ids)


class CocoImageRetrievalDataset(data.Dataset):
    """
    Custom COCO Dataset that uses only the images together with a user query.
    Compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, imgs_root, img_features_path, captions_json, coco_annotation_ids, query, num_imgs):
        self.query = query
        self.num_imgs = num_imgs
        self.feats_data_path = os.path.join(img_features_path, 'bu_att')
        self.box_data_path = os.path.join(img_features_path, 'bu_box')
        self.imgs_root = imgs_root

        self.coco = COCO(captions_json)
        self.anno_ids = coco_annotation_ids

    def __getitem__(self, idx):
        """
        This function returns a tuple that is further passed to collate_fn
        """
        img_id, img_size = self.get_raw_item(idx)

        img_feat_path = os.path.join(self.feats_data_path, '{}.npz'.format(img_id))
        img_box_path = os.path.join(self.box_data_path, '{}.npy'.format(img_id))

        img_feat = np.load(img_feat_path)['feat']
        img_feat_box = np.load(img_box_path)

        # normalize box
        img_feat_box = img_feat_box / np.tile(img_size, 2)

        img_feat = torch.Tensor(img_feat)
        img_feat_box = torch.Tensor(img_feat_box)

        # we always return the query here since we want to compute the similarity of each image with the query
        # this output is the input of the CollateFn
        return img_feat, img_feat_box, img_id, self.query, idx

    def get_raw_item(self, idx):
        next_img_idx = idx * 5  # in the coco dataset there are 5 captions for every image
        ann_id = self.anno_ids[next_img_idx]
        img_id = self.coco.anns[ann_id]['image_id']
        img_metadata = self.coco.imgs[img_id]
        img_size = np.array([img_metadata['width'], img_metadata['height']])

        return img_id, img_size

    def get_image_metadata(self, idx):
        # TODO can't we just get coco.imgs[idx'] somehow?
        next_img_idx = idx * 5  # in the coco dataset there are 5 captions for every image
        ann_id = self.anno_ids[next_img_idx]
        img_id = self.coco.anns[ann_id]['image_id']
        img_metadata = self.coco.imgs[img_id]
        return img_metadata

    def __len__(self):
        return self.num_imgs


class BottomUpFeaturesDataset:
    def __init__(self, imgs_root, captions_json, features_path, split, ids=None, **kwargs):
        # which dataset?
        r = imgs_root[0] if type(imgs_root) == tuple else imgs_root
        r = r.lower()
        if 'coco' in r:
            self.underlying_dataset = CocoDataset(imgs_root, captions_json, coco_annotation_ids=ids)
        elif 'f30k' in r or 'flickr30k' in r:
            self.underlying_dataset = FlickrDataset(imgs_root, captions_json, split)

        # data_path = config['image-model']['data-path']
        self.feats_data_path = os.path.join(features_path, 'bu_att')
        self.box_data_path = os.path.join(features_path, 'bu_box')
        config = kwargs['config']
        self.load_preextracted = config['text-model']['pre-extracted']
        if self.load_preextracted:
            # TODO: handle different types of preextracted features, not only BERT
            text_extractor = HuggingFaceTransformerExtractor(config, split, finetuned=config['text-model']['fine-tune'])
            self.text_features_db = FeatureSequence(text_extractor)

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        root, caption, img_id, _, _, img_size = self.underlying_dataset.get_raw_item(index, load_image=False)
        img_feat_path = os.path.join(self.feats_data_path, '{}.npz'.format(img_id))
        img_box_path = os.path.join(self.box_data_path, '{}.npy'.format(img_id))

        img_feat = np.load(img_feat_path)['feat']
        img_boxes = np.load(img_box_path)

        # normalize boxes
        img_boxes = img_boxes / np.tile(img_size, 2)

        img_feat = torch.Tensor(img_feat)
        img_boxes = torch.Tensor(img_boxes)

        if self.load_preextracted:
            record = self.text_features_db[index]
            features = record['features']
            captions = record['captions']
            wembeddings = record['wembeddings']
            target = (captions, features, wembeddings)
        else:
            target = caption
        # image = (img_feat, img_boxes)
        return img_feat, img_boxes, target, index, img_id  # target is the actual caption sentence

    def __len__(self):
        return len(self.underlying_dataset)


class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root, json, split, transform=None, get_images=True):
        self.root = root
        self.split = split
        self.get_images = get_images
        self.transform = transform
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]

        # dump flickr images sizes on files for later use
        size_file = os.path.join(root, 'sizes.pkl')
        if os.path.isfile(size_file):
            # load it
            with open(size_file, 'rb') as f:
                self.sizes = pickle.load(f)
        else:
            # build it
            sizes = []
            for im in tqdm.tqdm(self.dataset):
                path = im['filename']
                image = Image.open(os.path.join(root, path))
                sizes.append(image.size)

            with open(size_file, 'wb') as f:
                pickle.dump(sizes, f)
            self.sizes = sizes

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        root, caption, img_id, path, image, _ = self.get_raw_item(index, self.get_images)
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        target = caption
        return image, target, index, img_id

    def get_raw_item(self, index, load_image=True):
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        img_size = self.sizes[img_id]

        if load_image:
            path = self.dataset[img_id]['filename']
            image = Image.open(os.path.join(root, path)).convert('RGB')
            return root, caption, img_id, path, image, img_size
        else:
            return root, caption, img_id, None, None, img_size

    def __len__(self):
        return len(self.ids)


class InferenceCollate(object):
    def __new__(cls, *args, **kwargs):
        # we only need to compute this once so it gets stored in a static class variable
        cls.query_token_ids = None
        cls.query_length = None
        cls.img_feat_length = None
        cls.img_feat_dim = None
        cls.bboxes_length = None
        cls.bboxes_dim = None

        return super(InferenceCollate, cls).__new__(cls)

    def __init__(self, config):
        self.vocab_type = str(config['text-model']['name']).lower()
        self.create_query_batch = bool(config['image-retrieval']['create_query_batch'])
        if self.vocab_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(config['text-model']['pretrain'])
        else:
            raise ValueError("Currently only BERT Tokenizer is supported!")

    @classmethod
    def set_query_token_ids(cls, query_token_ids):
        cls.query_token_ids = query_token_ids
        cls.query_length = len(query_token_ids)

    @classmethod
    def set_img_feat_length_and_dimension(cls, img_feat):
        # +1 because the first region feature is reserved as CLS
        cls.img_feat_length = img_feat.shape[0] + 1
        cls.img_feat_dim = img_feat.shape[1]

    @classmethod
    def set_bboxes_length_and_dimension(cls, bbox):
        # +1 because the first region feature is reserved as CLS
        cls.bboxes_length = bbox.shape[0] + 1
        cls.bboxes_dim = bbox.shape[1]

    def __call__(self, data):
        img_feats, img_feat_bboxes, img_ids, queries, dataset_indices = zip(*data)
        """
        Build batch tensors from a list of (img_feats, img_feat_boxes, img_ids, queries, dataset_indices) tuples.
            Args:
                - img_feats:
                - img_feat_bboxes:
                - img_ids:
                - queries:
                - dataset_indices:

            Returns:
                - img_feature_batch: batch of image features
                - img_feat_bboxes_batch: batch of bounding boxes of the image features
                - img_feat_length: length of the image features and bounding boxes (all of same size)
                - query_token_ids: bert token ids of the tokenized query
                - query_length: length of the query
                - dataset_indices: indices of the elements of the datasets inside the batch.
        """

        # encode (tokenize) the query
        if self.query_token_ids is None:
            # we don't need to pad or truncate since we only have a single query
            # TODO actually we don't even need the tokenizer twice so we could just use a local variable
            query_token_ids = torch.LongTensor(self.tokenizer.encode(queries[0]))
            self.set_query_token_ids(query_token_ids)

        # prepare image features
        if self.img_feat_length is None:
            self.set_img_feat_length_and_dimension(img_feats[0])

        # prepare bounding boxes
        if self.bboxes_length is None:
            self.set_bboxes_length_and_dimension(img_feat_bboxes[0])

        assert self.bboxes_length == self.img_feat_length

        # create the image feature batch
        batch_size = len(img_feats)
        img_feature_batch = torch.zeros(batch_size, self.img_feat_length, self.img_feat_dim)
        for i, f in enumerate(img_feats):
            # reserve the first token as CLS
            img_feature_batch[i, 1:] = f

        # create the image features bounding boxes batch
        img_feat_lengths = [self.img_feat_length for _ in range(batch_size)]
        img_feat_bboxes_batch = torch.zeros(batch_size, self.bboxes_length, self.bboxes_dim)
        for i, box in enumerate(img_feat_bboxes):
            img_feat_bboxes_batch[i, 1:] = box

        if self.create_query_batch:
            # create the full query batch of size B x |Q|
            # since the token id is a scalar, the dim is 1 and whe don't need to add it to the batch
            # for the BERT embeddings the ids have to be Long
            query_token_ids_batch = torch.zeros(batch_size, self.query_length).long()
            for i in range(len(queries)):
                query_token_ids_batch[i] = self.query_token_ids
            query_lengths = [self.query_length for _ in range(batch_size)]
        else:
            # create a pseudo query batch with only one element of size 1 x |Q|
            query_token_ids_batch = self.query_token_ids.unsqueeze(dim=0)
            query_lengths = [self.query_length]

        return img_feature_batch, img_feat_bboxes_batch, img_feat_lengths, query_token_ids_batch, query_lengths, dataset_indices


class Collate:
    def __init__(self, config):
        self.vocab_type = config['text-model']['name']
        if self.vocab_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(config['text-model']['pretrain'])

    def __call__(self, data):
        """Build mini-batch tensors from a list of (image, caption) tuples.
            Args:
                data: list of (image, caption) tuple.
                    - image: torch tensor of shape (3, 256, 256) or (? > 3, 2048)
                    - caption: torch tensor of shape (?); variable length.

            Returns:
                images: torch tensor of shape (batch_size, 3, 256, 256).
                targets: torch tensor of shape (batch_size, padded_length). -> the textual tokens
                lengths: list; valid length for each padded caption.
            """
        # Sort a data list by caption length
        # data.sort(key=lambda x: len(x[1]), reverse=True)
        if len(data[0]) == 5:  # TODO: find a better way to distinguish the two
            images, boxes, captions, ids, img_ids = zip(*data)
        elif len(data[0]) == 4:
            images, captions, ids, img_ids = zip(*data)

        preextracted_captions = type(captions[0]) is tuple
        if preextracted_captions:
            # they are pre-extracted features
            captions, cap_features, wembeddings = zip(*captions)
            cap_lengths = [len(cap) for cap in cap_features]
            captions = [torch.LongTensor(c) for c in captions]
            cap_features = [torch.FloatTensor(f) for f in cap_features]
            wembeddings = [torch.FloatTensor(w) for w in wembeddings]
        else:
            if self.vocab_type == 'bert': 
                cap_lengths = [len(self.tokenizer.tokenize(c)) + 2 for c in
                               captions]  # + 2 in order to account for begin and end tokens
                max_len = max(cap_lengths)
                captions_token_ids = [torch.LongTensor(self.tokenizer.encode(c,
                                                                             max_length=max_len,
                                                                             padding='max_length',
                                                                             truncation=True))
                                      for c in captions]

            captions = captions_token_ids  # caption_ids are the token ids from bert tokenizer
        # Merge images (convert tuple of 3D tensor to 4D tensor)
        preextracted_images = not (images[0].shape[0] == 3)
        if not preextracted_images:
            # they are images
            images = torch.stack(images, 0)
        else:
            # they are image features, variable length
            feat_lengths = [f.shape[0] + 1 for f in images]  # +1 because the first region feature is reserved as CLS
            feat_dim = images[0].shape[1]
            img_features = torch.zeros(len(images), max(feat_lengths), feat_dim)
            for i, img in enumerate(images):
                end = feat_lengths[i]
                img_features[i, 1:end] = img

            box_lengths = [b.shape[0] + 1 for b in boxes]  # +1 because the first region feature is reserved as CLS
            assert box_lengths == feat_lengths
            out_boxes = torch.zeros(len(boxes), max(box_lengths), 4)
            for i, box in enumerate(boxes):
                end = box_lengths[i]
                out_boxes[i, 1:end] = box

        # Merget captions (convert tuple of 1D tensor to 2D tensor)
        if preextracted_captions:
            captions_t = torch.zeros(len(captions), max(cap_lengths)).long()
            features_t = torch.zeros(len(cap_features), max(cap_lengths), cap_features[0].shape[1])
            wembeddings_t = torch.zeros(len(wembeddings), max(cap_lengths), wembeddings[0].shape[1])
            for i, (cap, feats, wembs, l) in enumerate(zip(captions, cap_features, wembeddings, cap_lengths)):
                captions_t[i, :l] = cap[:l]
                features_t[i, :l] = feats[:l]
                wembeddings_t[i, :l] = wembs[:l]
            targets = (captions_t, features_t, wembeddings_t)
        else:
            targets = torch.zeros(len(captions), max(cap_lengths)).long()
            for i, cap in enumerate(captions):
                end = cap_lengths[i]
                targets[i, :end] = cap[:end]  #caption token ids

        if not preextracted_images:
            return images, targets, None, cap_lengths, None, ids
        else:
            # features = features.permute(0, 2, 1)
            # img_features -> from FRCNN >> B x 2048
            # targets -> padded caption token ids from BERT >> B x max_len(cap_lengths) or(queries)
            # feat_lengths -> num of regions in the image (fixed to 36 + 1) >> B x 37
            # cap_lengths -> true length of the non-padded captions or queries >> B x 1 (list of len B)
            # out_boxes -> spatial information of the region boxes >> B x 37 x 4
            # ids -> dataset indices wich are in this batch >> 1 x B (tuple of len B)
            return img_features, targets, feat_lengths, cap_lengths, out_boxes, ids


def get_loader_single(data_name, split, imgs_root, captions_json, transform, pre_extracted_root=None,
                      batch_size=100, shuffle=True,
                      num_workers=2, ids=None, collate_fn=None, **kwargs):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if 'coco' in data_name:
        if pre_extracted_root is not None:
            dataset = BottomUpFeaturesDataset(imgs_root=imgs_root,
                                              captions_json=captions_json,
                                              features_path=pre_extracted_root, split=split,
                                              ids=ids, **kwargs)
        else:
            # COCO custom dataset
            dataset = CocoDataset(imgs_root=imgs_root,
                                  captions_json=captions_json,
                                  transform=transform, coco_annotation_ids=ids)
    elif 'f8k' in data_name or 'f30k' in data_name:
        if pre_extracted_root is not None:
            dataset = BottomUpFeaturesDataset(imgs_root=imgs_root,
                                              captions_json=captions_json,
                                              features_path=pre_extracted_root, split=split,
                                              ids=ids, **kwargs)
        else:
            dataset = FlickrDataset(root=imgs_root,
                                    split=split,
                                    json=captions_json,
                                    transform=transform)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


def get_transform(data_name=None, split_name=None, config=None):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    # if split_name == 'train':
    #     t_list = [transforms.RandomResizedCrop(config['image-model']['crop-size']),
    #               transforms.RandomHorizontalFlip()]
    # elif split_name == 'val':
    #     t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    # elif split_name == 'test':
    #     t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform


def get_loaders(config, workers, batch_size=None):
    data_name = config['dataset']['name']
    if batch_size is None:
        batch_size = config['training']['bs']
    collate_fn = Collate(config)
    roots, ids = get_paths(config)

    transform = get_transform(data_name, 'train', config)
    preextracted_root = config['image-model']['pre-extracted-features-root'] \
        if 'pre-extracted-features-root' in config['image-model'] else None

    train_loader = get_loader_single(data_name, 'train',
                                     roots['train']['img'],
                                     roots['train']['cap'],
                                     transform, ids=ids['train'],
                                     pre_extracted_root=preextracted_root,
                                     batch_size=batch_size, shuffle=True,
                                     num_workers=workers,
                                     collate_fn=collate_fn, config=config)

    transform = get_transform(data_name, 'val', config)
    val_loader = get_loader_single(data_name, 'val',
                                   roots['val']['img'],
                                   roots['val']['cap'],
                                   transform, ids=ids['val'],
                                   pre_extracted_root=preextracted_root,
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=workers,
                                   collate_fn=collate_fn, config=config)

    return train_loader, val_loader


def get_coco_image_retrieval_data_loader(config, workers, query):
    # create the dataset + loader
    # 1) load / create a Coco Dataset to get meta info about images (we could also do this by hand)
    # 2) choose (the first) N images and create a dataset with N samples where each sample consists of the n-th image
    #    and the query (gets repeated N times) # TODO maybe this is not necessary

    # get the directories that contain the coco json files and coco annotation ids (which we may not need, I think)
    roots, coco_annotation_ids = get_paths(config)

    dataset_name = config['image-retrieval']['dataset']
    batch_size = config['image-retrieval']['batch_size']
    split_name = config['image-retrieval']['split']

    imgs_root = roots[split_name]['img']

    # for images we use pre-extracted features (not for text)
    pre_extracted_img_features_root = config['image-retrieval']['pre_extracted_img_features_root']

    captions_json = roots[split_name]['cap']
    coco_annotation_ids = coco_annotation_ids[split_name]
    num_imgs = config['image-retrieval']['num_imgs']

    dataset = CocoImageRetrievalDataset(imgs_root=imgs_root,
                                        img_features_path=pre_extracted_img_features_root,
                                        captions_json=captions_json,
                                        coco_annotation_ids=coco_annotation_ids,
                                        query=query,
                                        num_imgs=num_imgs)

    # this creates the batches which get passed to the model (inside the query gets repeated or not based on the config)
    collate_fn = InferenceCollate(config)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=workers,
                                              collate_fn=collate_fn)

    return data_loader


def get_test_loader(config, workers, split_name='test', batch_size=None):
    data_name = config['dataset']['name']
    if batch_size is None:
        batch_size = config['training']['bs']
    collate_fn = Collate(config)
    # Build Dataset Loader
    roots, ids = get_paths(config)

    pre_extracted_root = config['image-model']['pre-extracted-features-root'] \
        if 'pre-extracted-features-root' in config['image-model'] else None

    transform = get_transform(data_name, split_name, config)
    test_loader = get_loader_single(data_name, split_name,
                                    imgs_root=roots[split_name]['img'],
                                    captions_json=roots[split_name]['cap'],
                                    transform=transform, ids=ids[split_name],
                                    pre_extracted_root=pre_extracted_root,
                                    batch_size=batch_size, shuffle=False,
                                    num_workers=workers,
                                    collate_fn=collate_fn, config=config)
    return test_loader
