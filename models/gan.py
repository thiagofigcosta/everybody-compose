import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import ConcatPrev
from models import vanilla_rnn


# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, num_notes, seq_len, embed_dim, dense_hidden_size, dropout=0.1):
        super(Discriminator, self).__init__()
        self.num_notes = num_notes
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        
        self.embeddings = nn.Embedding(num_notes, embed_dim)
        
        kernel_size_0 = 16
        n_filters_0 = 100
        stride_0 = 2
        
        kernel_size_1 = 8
        n_filters_1 = 50
        stride_1 = 1
        
        kernel_size_2 = 8
        n_filters_2 = 50
        stride_2 = 1
        
        dense_input_size = 900

        self.model = nn.Sequential(
            # conv group 1
            nn.Conv2d(seq_len, n_filters_0, kernel_size=(kernel_size_0, embed_dim), stride=stride_0, padding=1, bias=False),
            nn.BatchNorm2d(n_filters_0),
            nn.LeakyReLU(dropout, inplace=True),
            nn.AvgPool2d(2), 
            
            # conv group 2
            nn.Conv2d(n_filters_0, n_filters_1, kernel_size=(kernel_size_1,1), stride=stride_1, padding=1, bias=False),
            nn.BatchNorm2d(n_filters_1),
            nn.LeakyReLU(dropout, inplace=True),
            
            # conv group 3
            nn.Conv2d(n_filters_1, n_filters_2, kernel_size=(kernel_size_2,1), stride=stride_2, padding=1, bias=False),
            nn.BatchNorm2d(n_filters_2),
            nn.LeakyReLU(dropout, inplace=True),
            nn.AvgPool2d(2), 
            
            # classify as fake or real
            nn.Flatten(),
            nn.Linear(dense_input_size, dense_hidden_size),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, beats, notes):
        X = torch.cat((beats, notes), dim=2)
        X = self.embeddings(X.long()).unsqueeze(1)
        pred = self.model(X.squeeze())
        return pred

# Generator Model
class Generator(vanilla_rnn.DeepBeatsVanillaRNN):
    pass