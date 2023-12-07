import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import ConcatPrev


# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, num_notes, seq_len, embed_dim, filter_sizes=[2, 3, 4, 5], num_filters=[300, 300, 300, 300], dropout=0.5):
        super(Discriminator, self).__init__()
        self.num_notes = num_notes
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.feature_dim = sum(num_filters)

        self.embeddings = nn.Linear(num_notes + 2, embed_dim, bias=True)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, embed_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])

        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, beats, notes):
        X = torch.cat((beats, notes), dim=2)
        X = self.embeddings(X).unsqueeze(1)
        convs = [F.relu(conv(X)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
        pred = torch.cat(pools, 1)
        highway = self.highway(pred)
        highway = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred
        pred = self.feature2out(self.dropout(highway))
        pred = torch.sigmoid(pred)
        return pred

    # def __init__(self, num_notes, seq_len, embed_dim, filter_sizes = [64, 32, 16], num_filters = [32, 16, 8], dropout = 0.1):
    #     super(Discriminator, self).__init__()

    #     self.num_notes = num_notes
    #     self.embed_dim = embed_dim
    #     self.seq_len = seq_len
    #     self.feature_dim = sum(num_filters)

    #     self.embeddings = nn.Linear(self.num_notes + 2, self.embed_dim, bias = False)

    #     self.model = nn.Sequential(
            
    #         nn.Conv2d(1, num_filters[0], kernel_size=filter_sizes[0], stride=3, padding='valid'),
    #         nn.ReLU(),
    #         # nn.AvgPool2d(2),
    #         nn.BatchNorm2d(num_filters[0], momentum=0.9),
    #         nn.Dropout(dropout),
            
    #         nn.Conv2d(num_filters[1-1], num_filters[1], kernel_size=filter_sizes[1], stride=2, padding='valid'),
    #         nn.ReLU(),
    #         nn.BatchNorm2d(num_filters[1], momentum=0.9),
    #         nn.Dropout(dropout),
            
    #         nn.Conv2d(num_filters[2-1], num_filters[2], kernel_size=filter_sizes[2], stride=1, padding='valid'),
    #         nn.ReLU(),
    #         nn.BatchNorm2d(num_filters[2], momentum=0.9),
    #         nn.Dropout(dropout),
            
    #         nn.Flatten(),
            
    #         nn.Linear(self.feature_dim * 8, 1024),
    #         nn.LeakyReLU(0.01),
    #         nn.BatchNorm2d(1024, momentum=0.9),
            
    #         nn.Linear(1024, 1),
    #         nn.Sigmoid()
    #     )

    # def forward(self, beats, notes):
    #     X = torch.cat((beats, notes), dim = 2)
    #     X = self.embeddings(X).unsqueeze(1)
    #     return self.model(X)


# Generator Model
class Generator(nn.Module):
    def __init__(self, num_notes, embed_size, hidden_dim):
        super(Generator, self).__init__()
        self.num_notes = num_notes
        self.note_embedding = nn.Embedding(num_notes, embed_size)
        self.concat_prev = ConcatPrev()
        self.concat_input_fc = nn.Linear(embed_size + 2, embed_size + 2)
        self.concat_input_activation = nn.LeakyReLU()
        self.layer1 = nn.RNN(embed_size + 2, hidden_dim, batch_first=True)
        self.layer2 = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        self.notes_output = nn.Linear(hidden_dim, num_notes)

        self._initializer_weights()

    def _default_init_hidden(self, batch_size):
        device = next(self.parameters()).device
        h1_0 = torch.zeros(1, batch_size, self.layer1.hidden_size).to(device)
        h2_0 = torch.zeros(1, batch_size, self.layer2.hidden_size).to(device)
        return h1_0, h2_0

    def _initializer_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y_prev, init_hidden = None):
        h1_0, h2_0 = self._default_init_hidden(x.shape[0]) if init_hidden is None else init_hidden
        y_prev_embed = self.note_embedding(y_prev)
        X = self.concat_prev(x, y_prev_embed)
        # Concat input
        X_fc = self.concat_input_fc(X)
        X_fc = self.concat_input_activation(X_fc)
        # residual connection
        X = X_fc + X
        X, h1 = self.layer1(X, h1_0)
        X, h2 = self.layer2(X, h2_0)
        predicted_notes = self.notes_output(X)
        return predicted_notes, (h1, h2)

    def loss_function(self, pred, target):
        criterion = nn.CrossEntropyLoss()
        target = target.flatten() # (batch_size * seq_len)
        pred = pred.reshape(-1, pred.shape[-1]) # (batch_size * seq_len, num_notes)
        loss = criterion(pred, target)
        return loss

    def clip_gradients_(self, max_value):
        torch.nn.utils.clip_grad.clip_grad_value_(self.parameters(), max_value)
