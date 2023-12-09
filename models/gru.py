import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import ConcatPrev

class DeepBeatsGRU(nn.Module):
    DOUBLE_GRU = True
    
    def __init__(self, num_notes, embed_size, hidden_dim, n_layers, dropout):
        super(DeepBeatsGRU, self).__init__()
        self.num_notes = num_notes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.note_embedding = nn.Embedding(num_notes, embed_size)
        self.concat_prev = ConcatPrev()
        self.concat_input_fc = nn.Linear(embed_size + 2, embed_size + 2)
        self.concat_input_activation = nn.LeakyReLU()
        
        self.gru = nn.GRU(embed_size + 2, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        if DeepBeatsGRU.DOUBLE_GRU:
            self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.notes_output = nn.Linear(hidden_dim, num_notes)

        self._initializer_weights()
        
    def forward(self, x, y_prev, init_hidden = None):
        hiddens = self.init_hidden(x.shape[0]) if init_hidden is None else init_hidden
        if DeepBeatsGRU.DOUBLE_GRU:
            hidden_0, hidden_2_0 = hiddens
        else:
            hidden_0 = hiddens
        y_prev_embed = self.note_embedding(y_prev)
        X = self.concat_prev(x, y_prev_embed)
        # Concat input
        X_fc = self.concat_input_fc(X)
        X_fc = self.concat_input_activation(X_fc)
        # residual connection
        X = X_fc + X
        X, hidden = self.gru(X, hidden_0)
        if DeepBeatsGRU.DOUBLE_GRU:
            X, hidden_2 = self.gru2(X, hidden_2_0)
        predicted_notes = self.notes_output(X)
        
        if DeepBeatsGRU.DOUBLE_GRU:
            return predicted_notes, (hidden, hidden_2)
        return predicted_notes, hidden
    
    def _initializer_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        weight = next(self.parameters()).data
        hidden_0 = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        if DeepBeatsGRU.DOUBLE_GRU:
            hidden_2_0 = weight.new(self.n_layers-1, batch_size, self.hidden_dim).zero_().to(device)
            return hidden_0, hidden_2_0
        return hidden_0
    
    def loss_function(self, pred, target):
        criterion = nn.CrossEntropyLoss()
        target = target.flatten() # (batch_size * seq_len)
        pred = pred.reshape(-1, pred.shape[-1]) # (batch_size * seq_len, num_notes)
        loss = criterion(pred, target)
        return loss

    def clip_gradients_(self, max_value):
        torch.nn.utils.clip_grad.clip_grad_value_(self.parameters(), max_value)


class DeepBeatsLSTM(nn.Module):
    def __init__(self, num_notes, embed_size, hidden_dim, n_layers, dropout):
        super(DeepBeatsLSTM, self).__init__()
        self.num_notes = num_notes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.note_embedding = nn.Embedding(num_notes, embed_size)
        self.concat_prev = ConcatPrev()
        self.concat_input_fc = nn.Linear(embed_size + 2, embed_size + 2)
        self.concat_input_activation = nn.LeakyReLU()
        
        self.lstm = nn.LSTM(embed_size + 2, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.notes_output = nn.Linear(hidden_dim, num_notes)

        self._initializer_weights()
        
    def forward(self, x, y_prev, init_hidden = None):
        hidden_0 = self.init_hidden(x.shape[0]) if init_hidden is None else init_hidden
        y_prev_embed = self.note_embedding(y_prev)
        X = self.concat_prev(x, y_prev_embed)
        # Concat input
        X_fc = self.concat_input_fc(X)
        X_fc = self.concat_input_activation(X_fc)
        # residual connection
        X = X_fc + X
        X, hidden = self.lstm(X, hidden_0)
        predicted_notes = self.notes_output(X)
        return predicted_notes, hidden

    def _initializer_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden
    
    def loss_function(self, pred, target):
        criterion = nn.CrossEntropyLoss()
        target = target.flatten() # (batch_size * seq_len)
        pred = pred.reshape(-1, pred.shape[-1]) # (batch_size * seq_len, num_notes)
        loss = criterion(pred, target)
        return loss

    def clip_gradients_(self, max_value):
        torch.nn.utils.clip_grad.clip_grad_value_(self.parameters(), max_value)