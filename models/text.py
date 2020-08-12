import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.utils import l2norm
from transformers import BertTokenizer, BertModel, BertConfig


def EncoderText(config):
    use_abs = config['training']['measure'] == 'order'
    num_layers = config['text-model']['layers']
    order_embeddings = config['training']['measure'] == 'order'
    if config['text-model']['name'] == 'gru':
        print('Using GRU text encoder')
        vocab_size = config['text-model']['vocab-size']
        word_dim = config['text-model']['word-dim']
        embed_size = config['model']['embed-size']
        model = EncoderTextGRU(vocab_size, word_dim, embed_size, num_layers, order_embeddings=order_embeddings)
    elif config['text-model']['name'] == 'bert':
        print('Using BERT text encoder')
        model = EncoderTextBERT(config, order_embeddings=order_embeddings, post_transformer_layers=num_layers)
    return model


# tutorials/08 - Language Model
# RNN Based Language Model
class EncoderTextGRU(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 order_embeddings=False):
        super(EncoderTextGRU, self).__init__()
        self.order_embeddings = order_embeddings
        self.embed_size = embed_size
        self.vocab_size = vocab_size

        # word embedding
        self.word_embeddings = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)

        self.init_weights()

    def init_weights(self):
        self.word_embeddings.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.word_embeddings(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = (I.expand(x.size(0), 1, self.embed_size)-1).to(x.device)
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        # out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.order_embeddings:
            out = torch.abs(out)

        return out, padded[0]

    def get_finetuning_params(self):
        return []


class EncoderTextBERT(nn.Module):
    def __init__(self, config, order_embeddings=False, mean=True, post_transformer_layers=0):
        super().__init__()
        self.preextracted = config['text-model']['pre-extracted']
        bert_config = BertConfig.from_pretrained(config['text-model']['pretrain'],
                                                 output_hidden_states=True,
                                                 num_hidden_layers=config['text-model']['extraction-hidden-layer'])
        bert_model = BertModel.from_pretrained(config['text-model']['pretrain'], config=bert_config)
        self.order_embeddings = order_embeddings
        self.vocab_size = bert_model.config.vocab_size
        self.hidden_layer = config['text-model']['extraction-hidden-layer']
        if not self.preextracted:
            self.tokenizer = BertTokenizer.from_pretrained(config['text-model']['pretrain'])
            self.bert_model = bert_model
            self.word_embeddings = self.bert_model.get_input_embeddings()
        if post_transformer_layers > 0:
            transformer_layer = nn.TransformerEncoderLayer(d_model=config['text-model']['word-dim'], nhead=4,
                                                           dim_feedforward=2048,
                                                           dropout=config['text-model']['dropout'], activation='relu')
            self.transformer_encoder = nn.TransformerEncoder(transformer_layer,
                                                             num_layers=post_transformer_layers)
        self.post_transformer_layers = post_transformer_layers
        self.map = nn.Linear(config['text-model']['word-dim'], config['model']['embed-size'])
        self.mean = mean

    def forward(self, x, lengths):
        '''
        x: tensor of indexes (LongTensor) obtained with tokenizer.encode() of size B x ?
        lengths: tensor of lengths (LongTensor) of size B
        '''
        if not self.preextracted or self.post_transformer_layers > 0:
            max_len = max(lengths)
            attention_mask = torch.ones(x.shape[0], max_len)
            for e, l in zip(attention_mask, lengths):
                e[l:] = 0
            attention_mask = attention_mask.to(x.device)

        if self.preextracted:
            outputs = x
        else:
            outputs = self.bert_model(x, attention_mask=attention_mask)
            outputs = outputs[2][-1]

        if self.post_transformer_layers > 0:
            outputs = outputs.permute(1, 0, 2)
            outputs = self.transformer_encoder(outputs, src_key_padding_mask=(attention_mask - 1).bool())
            outputs = outputs.permute(1, 0, 2)
        if self.mean:
            x = outputs.mean(dim=1)
        else:
            x = outputs[:, 0, :]     # from the last layer take only the first word

        out = self.map(x)

        # normalization in the joint embedding space
        # out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.order_embeddings:
            out = torch.abs(out)
        return out, outputs

    def get_finetuning_params(self):
        return list(self.bert_model.parameters())
