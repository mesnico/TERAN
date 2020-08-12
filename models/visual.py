from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
from torchvision import models as models
from models.utils import PositionalEncodingImageBoxes, l2norm


def EncoderImage(config):

    # data_name, img_dim, embed_size, finetune=False,
    #         cnn_type='vgg19', use_abs=False, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an encoder that uses
    precomputed image features, `EncoderImagePrecomp`, or an encoder that
    computes image features on the fly `EncoderImageFull`.
    """

    embed_size = config['model']['embed-size']
    order_embeddings = config['training']['measure'] == 'order'
    no_imgnorm = not config['image-model']['norm']
    if config['dataset']['pre-extracted-features']:
        img_dim = config['image-model']['feat-dim']
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, order_embeddings, no_imgnorm)
    else:
        if config['image-model']['name'] == 'cnn':
            finetune = config['image-model']['fine-tune']
            cnn_type = config['image-model']['model']
            use_transformer = config['image-model']['use-transformer']
            img_enc = EncoderImageFull(
                embed_size, finetune, cnn_type, order_embeddings, no_imgnorm, use_transformer=use_transformer)
        elif config['image-model']['name'] == 'bottomup':
            transformer_layers = config['image-model']['transformer-layers']
            pos_encoding = config['image-model']['pos-encoding']
            visual_feat_dim = config['image-model']['feat-dim']
            dropout = config['image-model']['dropout']
            img_enc = TransformerPostProcessing(transformer_layers, visual_feat_dim, embed_size, n_head=4, aggr='mean', pos_encoding=pos_encoding, dropout=dropout, order_embeddings=order_embeddings)
        elif config['image-model']['name'] == 'gcn':
            img_dim = config['image-model']['feat-dim']
            img_enc = GCNVisualReasoning(img_dim, embed_size, data_name='coco', use_abs = False, no_imgnorm = False)
        else:
            img_enc = None

    return img_enc


class TransformerPostProcessing(nn.Module):
    def __init__(self, num_transformer_layers, feat_dim, embed_size, n_head=4, aggr='mean', pos_encoding=None, dropout=0.1, order_embeddings=False):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=n_head,
                                                       dim_feedforward=2048,
                                                       dropout=dropout, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer,
                                                         num_layers=num_transformer_layers)
        if pos_encoding is not None:
            self.pos_encoding_image = PositionalEncodingImageBoxes(feat_dim, pos_encoding)
        self.fc = nn.Linear(feat_dim, embed_size)
        self.aggr = aggr
        self.order_embeddings = order_embeddings
        if aggr == 'gated':
            self.gate_fn = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, 1)
            )
            self.node_fn = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim)
            )
        self.pos_encoding = pos_encoding

    def forward(self, visual_feats, visual_feats_len=None, boxes=None):
        """
        Takes an variable len batch of visual features and preprocess them through a transformer. Output a tensor
        with the same shape as visual_feats passed in input.
        :param visual_feats:
        :param visual_feats_len:
        :return: a tensor with the same shape as visual_feats passed in input.
        """
        # max_len = max(visual_feats_len)
        # bs = visual_feats.shape[1]
        # attention_mask = torch.zeros(bs, max_len).bool()
        # for e, l in zip(attention_mask, visual_feats_len):
        #     e[l:] = True
        # attention_mask = attention_mask.to(visual_feats.device)

        visual_feats = visual_feats.permute(1, 0, 2)
        if self.pos_encoding is not None:
            visual_feats = self.pos_encoding_image(visual_feats, boxes)

        if visual_feats_len is not None:
            bs = visual_feats.shape[1]
            # construct the attention mask
            max_len = max(visual_feats_len)
            mask = torch.zeros(bs, max_len).bool()
            for e, l in zip(mask, visual_feats_len):
                e[l:] = True
            mask = mask.to(visual_feats.device)
        else:
            mask = None

        visual_feats = self.transformer_encoder(visual_feats, src_key_padding_mask=mask)
        # visual_feats = visual_feats.permute(1, 0, 2)

        if self.aggr == 'mean':
            out = visual_feats.mean(dim=0)
        elif self.aggr == 'gated':
            out = visual_feats.permute(1, 0, 2)
            m = torch.sigmoid(self.gate_fn(out))   # B x S x 1
            v = self.node_fn(out)   # B x S x dim
            out = torch.bmm(m.permute(0, 2, 1), v)      # B x 1 x dim
            out = out.squeeze(1)    # B x dim
        else:
            out = visual_feats[0]

        out = self.fc(out)
        if self.order_embeddings:
            out = torch.abs(out)

        return out, visual_feats.permute(1, 0, 2)


def find_nhead(feat_dim, higher=8):
    # find the right n_head value (the highest value lower than 'higher')
    for i in reversed(range(higher + 1)):
        if feat_dim % i == 0:
            return i
    return 1


class GCNVisualReasoning(nn.Module):

    def __init__(self, img_dim, embed_size, data_name, use_abs=False, no_imgnorm=False):
        super(GCNVisualReasoning, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.data_name = data_name

        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

        # GSR
        self.img_rnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)

        # GCN reasoning
        self.Rs_GCN_1 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_2 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_3 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_4 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)

        if self.data_name == 'f30k_precomp':
            self.bn = nn.BatchNorm1d(embed_size)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, img_len=None, boxes=None):
        assert not any(np.array(img_len) - img_len[0])
        """Extract image feature vectors."""

        fc_img_emd = self.fc(images)
        if self.data_name != 'f30k_precomp':
            fc_img_emd = l2norm(fc_img_emd)

        # GCN reasoning
        # -> B,D,N
        GCN_img_emd = fc_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_2(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_3(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_4(GCN_img_emd)
        # -> B,N,D
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)

        GCN_img_emd = l2norm(GCN_img_emd)

        rnn_img, hidden_state = self.img_rnn(GCN_img_emd)

        # features = torch.mean(rnn_img,dim=1)
        features = hidden_state[0]

        if self.data_name == 'f30k_precomp':
            features = self.bn(features)

            # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features, GCN_img_emd

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super().load_state_dict(new_state)


class EncoderImageFull(nn.Module):

    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 use_abs=False, no_imgnorm=False, avgpool_size=(4, 4), use_transformer=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type, True)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            raise NotImplementedError
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.spatial_feats_dim = self.cnn.module.fc.in_features
            modules = list(self.cnn.module.children())[:-2]
            self.cnn = torch.nn.Sequential(*modules)
            self.avgpool = nn.AdaptiveAvgPool2d(avgpool_size)
            self.glob_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.spatial_feats_dim, embed_size)

            # self.cnn.module.fc = nn.Sequential()

        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = TransformerPostProcessing(2, self.spatial_feats_dim, embed_size, n_head=4)
        self.init_weights()

    def get_cnn(self, arch, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]()

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
        else:
            model = nn.DataParallel(model)

        if torch.cuda.is_available():
            model.cuda()

        return model

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']

        super(EncoderImageFull, self).load_state_dict(state_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        spatial_features = self.cnn(images)
        features = self.glob_avgpool(spatial_features)   # compute a single feature
        spatial_features = self.avgpool(spatial_features)   # fix the size of the spatial grid

        if not self.use_transformer:
            features = torch.flatten(features, 1)
            # normalization in the image embedding space
            features = l2norm(features)
            # linear projection to the joint embedding space
            features = self.fc(features)
        else:
            # transformer + fc projection to the joint embedding space
            features, _ = self.transformer(spatial_features.view(spatial_features.shape[0], spatial_features.shape[1], -1).permute(2, 0, 1))

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features, spatial_features

    def get_finetuning_params(self):
        return list(self.cnn.parameters())


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)