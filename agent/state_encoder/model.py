
## eventually need to implement GPT-VAE to replace the below model
## https://arxiv.org/pdf/2205.05862.pdf


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class AttentionEncoderDecoder(pl.LightningModule):
    def __init__(self, vocab_size, emb_dim, feature_dim, seq_len, dropout=0.1, num_layers=1, lr=0.0001):
        super().__init__()
        self.lr = lr
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.encoder = Encoder(vocab_size, seq_len, emb_dim, feature_dim, dropout, num_layers)
        self.decoder = Decoder(vocab_size, seq_len, feature_dim, self.encoder.hidden_dim)

        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()

    def update_vocab(self, new_size):
        embedding_layer = self.encoder.embeddings
        old_num_tokens, old_embedding_dim = embedding_layer.weight.shape

        new_embeddings = nn.Embedding(new_size, old_embedding_dim, padding_idx=0)
        new_embeddings.to(
            embedding_layer.weight.device,
            dtype=embedding_layer.weight.dtype,
        )
        new_embeddings.weight.data[:old_num_tokens, :] = embedding_layer.weight.data[:old_num_tokens, :]
        self.encoder.embeddings = new_embeddings

        output = self.decoder.output_sequence.module
        new_output = TimeDistributed(nn.Linear(self.decoder.hidden_dim, new_size))

        new_output.module.weight.data[:old_num_tokens, :] = output.weight.data[:old_num_tokens, :]
        new_output.module.bias.data[:old_num_tokens] = output.bias.data[:old_num_tokens]

        self.decoder.output_sequence = new_output
        self.vocab_size = new_size
        self.decoder.vocab_size = new_size
        self.hparams['vocab_size'] = new_size
        self._hparams_initial['vocab_size'] = new_size

    def pad(self, x, pad_token, pad_side='right'):
        if pad_side == 'right':
            pad = (0, self.seq_len - len(x))
        elif pad_side == 'left':
            pad = (self.seq_len - len(x), 0)
        else:
            raise ValueError('choose left or right for pad_side')
        x = torch.tensor(np.pad(x, pad, mode='constant', constant_values=(pad_token, pad_token)))
        return x

    def encode(self, x, pad_token=0, pad_side='right'):
        x = self.pad(x, pad_token, pad_side)
        if len(x.size()) < 2:
            x = x.unsqueeze(0)

        x = self.encoder(x)
        return x

    def decode(self, x, mask=None):
        if torch.is_tensor(mask) and len(mask.size()) < 2:
            mask = mask.unsqueeze(0)

        x = self.decoder(x, mask)
        x = F.softmax(x, dim=1)
        x = x.argmax(2).squeeze()
        return x

    def forward(self, target, mask=None):
        x = self.encoder(target)
        y = self.decoder(x, mask)
        return y

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return opt

    def training_step(self, batch, batch_idx):
        obs = batch['obs']
        mask = batch['mask']
        logits = self(obs, mask)
        loss = self.criterion(logits.permute(0, 2, 1), obs)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        obs = batch['obs']
        mask = batch['mask']
        logits = self(obs, mask)
        loss = self.criterion(logits.permute(0, 2, 1), obs)
        proba = F.softmax(logits, dim=1)
        pred = proba.argmax(2)

        self.log('loss', loss, on_epoch=True, sync_dist=True)
        outputs = {'loss': loss, 'proba': proba, 'pred': pred}
        return outputs


class Encoder(nn.Module):
    def __init__(self, vocab_size, seq_len, emb_dim, feature_dim, dropout=0.1, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_dim = emb_dim // 2

        self.embeddings = nn.Embedding(self.vocab_size, emb_dim, padding_idx=0)
        self.attn = Attention(emb_dim)
        self.conv = nn.Conv2d(1, 1, kernel_size=(self.seq_len, 1))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(emb_dim, feature_dim)

    def forward(self, x):
        emb = self.embeddings(x)
        attend, _ = self.attn(emb, emb)
        conv = self.conv(attend.unsqueeze(1)).squeeze(1).squeeze(1)
        drop = self.dropout(conv)
        dense = self.linear(drop)
        return dense


class Decoder(nn.Module):
    def __init__(self, vocab_size, seq_len, feature_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim * 2

        self.linear = nn.Linear(feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.convt = nn.ConvTranspose1d(hidden_dim, hidden_dim * 2, self.seq_len)
        self.attn = Attention(hidden_dim * 2)
        self.output_sequence = TimeDistributed(nn.Linear(hidden_dim * 2, self.vocab_size))

    def forward(self, x, mask=None):
        dense = self.linear(x)
        drop = self.dropout(dense)
        drop_view = drop.view(drop.size(0), drop.size(1), 1)
        unconv = self.convt(drop_view)
        batch_view = unconv.view(unconv.size(0), unconv.size(2), unconv.size(1))
        attend, _ = self.attn(batch_view, batch_view)
        out = F.relu(attend)
        logits = self.output_sequence(out)

        if torch.is_tensor(mask):
            mask = mask.unsqueeze(2).expand(-1, -1, self.vocab_size)
            masked_logits = logits.masked_fill((1 - mask).bool(), float('-inf'))
            masked_logits[:, :, 0] = logits[:, :, 0]
            logits = masked_logits

        return logits


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        x_reshape = x.contiguous().view(-1, x.size(-1))

        y = self.module(x_reshape)

        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))
        return y


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights


class LSTMEncoderDecoder(pl.LightningModule):
    def __init__(self, vocab_size, emb_dim, feature_dim, seq_len, hidden_dim, dropout=0.1, num_layers=1, lr=0.0001):
        super().__init__()
        self.lr = lr
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.encoder = LSTMEncoder(self.vocab_size, emb_dim, feature_dim, seq_len, hidden_dim, dropout, num_layers)
        self.decoder = LSTMDecoder(self.vocab_size, feature_dim, seq_len, hidden_dim)

        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()

    def update_vocab(self, new_size):
        embedding_layer = self.encoder.embeddings
        old_num_tokens, old_embedding_dim = embedding_layer.weight.shape

        new_embeddings = nn.Embedding(new_size, old_embedding_dim, padding_idx=0)
        new_embeddings.to(
            embedding_layer.weight.device,
            dtype=embedding_layer.weight.dtype,
        )
        new_embeddings.weight.data[:old_num_tokens, :] = embedding_layer.weight.data[:old_num_tokens, :]
        self.encoder.embeddings = new_embeddings

        output = self.decoder.output_sequence.module
        new_output = TimeDistributed(nn.Linear(self.decoder.lstm_dim, new_size))

        new_output.module.weight.data[:old_num_tokens, :] = output.weight.data[:old_num_tokens, :]
        new_output.module.bias.data[:old_num_tokens] = output.bias.data[:old_num_tokens]

        self.decoder.output_sequence = new_output
        self.vocab_size = new_size
        self.decoder.vocab_size = new_size
        self.hparams['vocab_size'] = new_size
        self._hparams_initial['vocab_size'] = new_size

    def pad(self, x, pad_token, pad_side='left'):
        if pad_side == 'right':
            pad = (0, self.seq_len - len(x))
        elif pad_side == 'left':
            pad = (self.seq_len - len(x), 0)
        else:
            raise ValueError('choose left or right for pad_side')
        x = torch.tensor(np.pad(x, pad, mode='constant', constant_values=(pad_token, pad_token)))
        return x

    def encode(self, x, pad_token=0, pad_side='left'):
        x = self.pad(x, pad_token, pad_side)
        if len(x.size()) < 2:
            x = x.unsqueeze(0)

        x = self.encoder(x)
        return x

    def decode(self, x, mask=None):
        if torch.is_tensor(mask) and len(mask.size()) < 2:
            mask = mask.unsqueeze(0)

        x = self.decoder(x, mask)
        x = F.softmax(x, dim=1)
        x = x.argmax(2).squeeze()
        return x

    def forward(self, target, mask=None):
        x = self.encoder(target)
        y = self.decoder(x, mask)
        return y

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return opt

    def training_step(self, batch, batch_idx):
        obs = batch['obs']
        mask = batch['mask']
        logits = self(obs, mask)
        loss = self.criterion(logits.permute(0, 2, 1), obs)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        obs = batch['obs']
        mask = batch['mask']
        logits = self(obs, mask)
        loss = self.criterion(logits.permute(0, 2, 1), obs)
        proba = F.softmax(logits, dim=1)
        pred = proba.argmax(2)

        self.log('loss', loss, on_epoch=True, sync_dist=True)
        outputs = {'loss': loss, 'proba': proba, 'pred': pred}
        return outputs


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, feature_dim, seq_len, hidden_dim=32, dropout=0.1, num_layers=1):
        super().__init__()
        self.flat_dim = hidden_dim * 2 * seq_len
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(self.vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.flat_dim, feature_dim)

    def forward(self, x):
        emb = self.embeddings(x)
        series, hidden = self.lstm(emb)
        drop = self.dropout(series)
        flat = torch.flatten(drop, start_dim=1)
        dense = self.linear(flat)
        return dense


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, feature_dim, seq_len, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.lstm_dim = hidden_dim * 2

        self.linear = nn.Linear(feature_dim, self.lstm_dim * seq_len)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.lstm_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.output_sequence = TimeDistributed(nn.Linear(self.lstm_dim, self.vocab_size))

    def forward(self, x, mask=None):
        dense = self.linear(x)
        drop = self.dropout(dense)
        unflatten = drop.view(-1, self.seq_len, self.lstm_dim)
        out, _ = self.lstm(unflatten)
        out = F.relu(out)
        logits = self.output_sequence(out)

        # zero out all other token preds except for the padding token (0) if mask is True
        if torch.is_tensor(mask):
            mask = mask.unsqueeze(2).expand(-1, -1, self.vocab_size)
            masked_logits = logits.masked_fill((1 - mask).bool(), float("-inf"))
            masked_logits[:, :, 0] = logits[:, :, 0]
            logits = masked_logits

        return logits
