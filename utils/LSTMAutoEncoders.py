import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=384, n_layers=1):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim = embedding_dim

        self.rnn = nn.LSTM(
          input_size=n_features,
          hidden_size=self.embedding_dim,
          num_layers=n_layers,
          batch_first=True
        )

        self.fc = nn.Linear(in_features=seq_len*embedding_dim, out_features=embedding_dim)

    def forward(self, x):
        # x = x.reshape((1, self.seq_len, self.n_features))
        # print("encoder input", x.size())
        x, (hidden_n, _) = self.rnn(x)
        # print("encoder output", x.size(), "hidden_n", hidden_n[0].size())
        return hidden_n


class Decoder(nn.Module):
    def __init__(self, seq_len, embedding_dim=384, n_features=128, n_layers=1):
        super(Decoder, self).__init__()

        self.seq_len, self.embedding_dim = seq_len, embedding_dim
        self.n_features = n_features

        self.rnn = nn.LSTM(
          input_size=self.embedding_dim,
          hidden_size=self.n_features,
          num_layers=n_layers,
          batch_first=True
        )

        # self.output_layer = nn.Linear(self.n_features, self.n_features)

    def forward(self, x):
        # x = batch, 384
        # 460, 128
        # print("decoder input", x.size())
        # x = x.squeeze(0)
        # print("decoder sq input", x.size())


        # x = x.unsqueeze(0).repeat(self.seq_len, 1)
        # x = x.unsqueeze(0).repeat(self.seq_len, 1)
        x = x.unsqueeze(1)
        # print("decoder unsq :",x.size())

        x = x.repeat(1,self.seq_len, 1)  # from batch,embedding ==> batch,seq_len,embedding by repeating embedding by seq_len times
        
        # print("decoder unsq repeated :",x.size())

        # x = x.reshape((1, self.seq_len, self.embedding_dim))
        x, (hidden_n, _) = self.rnn(x)
        # x = x.reshape((self.seq_len, self.n_features))
        # print("decoder output:",x.size(), "decoder hidden ", hidden_n.size())
        return x


class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=384):
        super(RecurrentAutoencoder, self).__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.squeeze(0)
        decoded = self.decoder(encoded)
        return encoded, decoded