import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ChemicalVAE(nn.Module):
    def __init__(self, 
                 vocab_size=35,
                 enc_hidden_dim=292,
                 latent_dim=292,
                 emb_dim=128,
                 dec_hidden_dim=501,
                 dropout=0.1):
        super().__init__()
        self.encoder = Encoder(vocab_size, 
                               emb_dim, 
                               enc_hidden_dim, 
                               dropout)
        self.decoder = Decoder(self.encoder.embedding, 
                               dropout, 
                               latent_dim, 
                               dec_hidden_dim, 
                               vocab_size, 
                               emb_dim)
        self.h2mu = nn.Linear(enc_hidden_dim, latent_dim)
        self.h2logvar = nn.Linear(enc_hidden_dim, latent_dim)
        self.apply(weight_init)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, seq: torch.Tensor, seq_len: torch.Tensor):

        hidden = self.encoder(seq, seq_len)
        mu, logvar = self.h2mu(hidden), self.h2logvar(hidden)
        z = self.reparameterize(mu, logvar)
        y, rec_loss = self.decoder(z, seq, seq_len)
        
        return mu, logvar, rec_loss, y
    
    
class Encoder(nn.Module):
    def __init__(self, 
                 vocab_dim=35,
                 emb_dim=128,
                 enc_hidden_dim=292,
                 dropout=0.2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_dim, emb_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(emb_dim, enc_hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(enc_hidden_dim * 2, enc_hidden_dim)
    
    def forward(self, seq: torch.Tensor, seq_len: torch.Tensor):
        input_emb = self.emb_dropout(self.embedding(seq))
        packed_input = pack_padded_sequence(input_emb, 
                                            seq_len.tolist(), 
                                            batch_first=True, 
                                            enforce_sorted=False)
        _, hidden = self.gru(packed_input)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        hidden = self.fc(hidden)
        return hidden


class Decoder(nn.Module):
    def __init__(self, 
                 embedding: nn.Embedding,
                 emb_dropout=0.2,
                 latent_dim=292,
                 hidden_dim=501,
                 vocab_dim=35,
                 emb_dim=128):
        super().__init__()
        
        self.embedding = embedding
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.gru = nn.GRU(latent_dim+emb_dim, hidden_dim, batch_first=True)
        self.z2h = nn.Linear(latent_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, vocab_dim)
    
    def forward(self, z: torch.Tensor, seq: torch.Tensor, seq_len: torch.Tensor):
        input_emb = self.emb_dropout(self.embedding(seq))
        z_0 = z.unsqueeze(1).repeat(1, input_emb.size(1), 1)
        x_input = torch.cat([input_emb, z_0], dim=-1)
        packed_input = pack_padded_sequence(x_input, 
                                            seq_len.tolist(), 
                                            batch_first=True, 
                                            enforce_sorted=False)
        h_0 = self.z2h(z)
        h_0 = h_0.unsqueeze(0).repeat(1, 1, 1)
        output, _ = self.gru(packed_input, h_0)
        packed_output, _ = pad_packed_sequence(output, batch_first=True)
        y = self.output_head(packed_output)
        
        reconstruction_loss = F.cross_entropy(y[:, :-1].contiguous().view(-1, y.size(-1)),
                                            seq[:, 1:torch.max(seq_len).item()].contiguous().view(-1),
                                            ignore_index=0)
        return y, reconstruction_loss


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GRU):
        
        nn.init.orthogonal_(m.all_weights[0][0])
        nn.init.orthogonal_(m.all_weights[0][1])
        nn.init.zeros_(m.all_weights[0][2])
        nn.init.zeros_(m.all_weights[0][3])
        
    elif isinstance(m, nn.Embedding):
        nn.init.constant_(m.weight, 1)


if __name__ == '__main__':
    vocab_size = 41
    input_seq = torch.randint(0, vocab_size, (5, 120))
    input_seq_len = torch.randint(1, 120, (5,))
    print(f"input_seq_len: {input_seq_len.shape}")
    # Test Encoder
    encoder = Encoder(vocab_dim=vocab_size)
    x = torch.rand(5, 121, 128)
    output = encoder(input_seq, input_seq_len)
    print(f"Encoder output shape: {output.shape}")
    
    # Test Decoder
    decoder = Decoder(vocab_dim=vocab_size, embedding=encoder.embedding)
    z = torch.randn(5, 292)
    output = decoder(z, input_seq, input_seq_len)
    print(f"Decoder output shape: {output[0].shape}, reconstruction_loss: {output[1].item()}")
    
    # Test ChemicalVAE
    model = ChemicalVAE(vocab_size=vocab_size)
    output = model(input_seq, input_seq_len)
    print(f"ChemicalVAE output shape: {output[0].shape}")