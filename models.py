import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from textblob import TextBlob


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        features = self.resnet(images)  # (batch_size,2048,7,7)
        features = features.permute(0, 2, 3, 1)  # (batch_size,7,7,2048)
        features = features.view(features.size(0), -1, features.size(-1))  # (batch_size,49,2048)
        return features


# Bahdanau Attention
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()

        self.attention_dim = attention_dim

        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)

        self.A = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        '''
        features: output of Encoder with shape (batch_size,49,encoder_dim)
        hidden_state: with shape (batch_size, decoder_dim)
        '''
        u_hs = self.U(features)  # (batch_size,num_layers,attention_dim)
        w_ah = self.W(hidden_state)  # (batch_size,attention_dim)

        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1))  # (batch_size,num_layers,attemtion_dim)

        attention_scores = self.A(combined_states)  # (batch_size,num_layers,1)
        attention_scores = attention_scores.squeeze(2)  # (batch_size,num_layers)

        alpha = F.softmax(attention_scores, dim=1)  # (batch_size,num_layers)

        attention_weights = features * alpha.unsqueeze(2)  # (batch_size,num_layers,features_dim)
        attention_weights = attention_weights.sum(dim=1)  # (batch_size,num_layers)

        return alpha, attention_weights

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3, embedding_weights = None):
        super().__init__()

        # save the model param
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.embedding_weights = embedding_weights
        if self.embedding_weights is not None:
            print('using pretrained embedding weights')
            assert self.embedding_weights.size(1) == embed_size
            assert self.embedding_weights.size(0) == vocab_size
            self.embedding,_,_ = self._create_emb_layer(embedding_weights)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.drop = nn.Dropout(drop_prob)
        
    #Loads embedding layer with pre-trained embeddings
    def _create_emb_layer(self, weights_matrix, non_trainable=False):
        # weights_matrix: Tensor of size (vocab_dim, embed_dim)
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer, num_embeddings, embedding_dim
    
    def forward(self, features, captions):

        # vectorize the caption
        embeds = self.embedding(captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        # get the seq length to iterate
        seq_length = len(captions[0]) - 1  # Exclude the last one
        batch_size = captions.size(0)
        num_features = features.size(1)

        preds = torch.zeros(batch_size, seq_length, self.vocab_size)
        alphas = torch.zeros(batch_size, seq_length, num_features)

        for s in range(seq_length):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fcn(self.drop(h))

            preds[:, s] = output
            alphas[:, s] = alpha

        return preds, alphas
    
    def generate_caption(self, features, max_len=20, vocab=None):
        # Inference part
        # Given the image features generate the captions

        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        alphas = []

        # starting input
        word = torch.tensor(vocab.stoi['<SOS>']).view(1, -1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        word = word.to(device)
        embeds = self.embedding(word)

        captions = []

        for i in range(max_len):
            alpha, context = self.attention(features, h)

            # store the apla score
            alphas.append(alpha.cpu().detach().numpy())

            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            output = output.view(batch_size, -1)

            # select the word with most val
            predicted_word_idx = output.argmax(dim=1)

            # save the generated word
            captions.append(predicted_word_idx.item())

            # end if <EOS detected>
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break

            # send generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))

        # covert the vocab idx to words and return sentence
        return [vocab.itos[idx] for idx in captions], alphas

    def init_hidden_state(self, features):
        mean_features = features.mean(dim=1)
        h = self.init_h(mean_features)  # (batch_size, decoder_dim)
        c = self.init_c(mean_features)
        return h, c

class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3, embedding_weights = None):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            embedding_weights = embedding_weights
        )

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

def load_embedding_weights(vocab, embedding_model):
    """
        Loads embedding layer with pre-trained embeddings.
        :param embedding_model: pre-trained embeddings model
        :param vocab: our task vocabulary
    """
    stoi = vocab.stoi
    matrix_len = len(stoi)
    weights_matrix = np.zeros((matrix_len, embedding_model.dim))
    for s,i in stoi.items():
        # if token exists in glove vocab
        if s in embedding_model.stoi.keys():
            weights_matrix[i] = embedding_model[s]
        # if not, take the embeddings from close token (fix spelling)
        elif str(TextBlob(s).correct()) in embedding_model.stoi.keys():
            #print(s,str(TextBlob(s).correct()))
            weights_matrix[i] = embedding_model[str(TextBlob(s).correct())]
        # else initialize randomly
        else:
            weights_matrix[i] = np.random.uniform(low=-1,high=1,size=embedding_model.dim)
    return torch.from_numpy(weights_matrix)