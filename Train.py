from data import *
from models import *
import torchvision.transforms as T
from torchtext.vocab import GloVe # for pretrained model
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle

def save_model(model, num_epochs):
    path = "caption_model_E_"+str(num_epochs)+".torch"
    torch.save(model, path)


captions_file_path = "captions.txt"
karpathy_json_path = 'Karpathy_data.json'
BATCH_SIZE = 32
NUM_WORKER = 0

# define the transforms to be applied which needed for the pretrained CNN
transforms = T.Compose([
    T.Resize(226),
    T.RandomCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

#build vocab
vocab = build_vocab(captions_file_path=captions_file_path)

#build datasets
train,val,test = karpathy_split(captions_file_path, karpathy_json_path)
train_dataset = FlickrDataset(root_dir='./Images',vocab= vocab, captions_df=train,transform=transforms)
val_dataset = FlickrDataset(root_dir='./Images',vocab= vocab, captions_df=val,transform=transforms)
test_dataset = FlickrDataset(root_dir='./Images',vocab= vocab, captions_df=test,transform=transforms)
print("Finished building the Datasets.")


# Hyperparams
weights_matrix = None
# load pretrained embeddings (to train embeddings from scrach just set the embed_size)
g = GloVe(name ='6B', dim=100) 
embed_size = g.dim
weights_matrix = load_embedding_weights(vocab, g)
 
vocab_size = len(vocab)
attention_dim = 256
encoder_dim = 2048
decoder_dim = 512
learning_rate = 3e-4


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#init model
model = EncoderDecoder(
    embed_size=embed_size,
    vocab_size=vocab_size,
    attention_dim=attention_dim,
    encoder_dim=encoder_dim,
    decoder_dim=decoder_dim,
    embedding_weights = weights_matrix
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"]).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10
print_every = 10

pad_idx = vocab.stoi["<PAD>"]

data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=True,
    collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True)
)

loss_list = []
perplexity_list = []
total_loss = 0
for epoch in range(1, num_epochs + 1):
    for idx, (image, captions) in enumerate(iter(data_loader)):
        image, captions = image.to(device), captions.to(device)

        # Zero the gradients.
        optimizer.zero_grad()

        # Feed forward
        outputs, attentions = model(image, captions)
        outputs = outputs.to(device)

        # Calculate the batch loss.
        targets = captions[:, 1:]
        targets = targets.to(device)
        loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))

        # Backward pass.
        loss.backward()

        # Update the parameters in the optimizer.
        optimizer.step()

        total_loss += loss.item()

        loss_list.append(loss.item())
        if (idx + 1) % print_every == 0:
            perplexity = total_loss / print_every
            perplexity = np.exp(perplexity)
            perplexity_list.append(perplexity)
            # generate the caption
            model.eval()
            with torch.no_grad():
                dataiter = iter(data_loader)
                img, true_caption = next(dataiter)
                true_caption = true_caption[0:1]
                features = model.encoder(img[0:1].to(device))
                caps, alphas = model.decoder.generate_caption(features, vocab=vocab)
                caption = ' '.join(caps)
                show_image(img[0], title=caption)
            print("Epoch: {} loss: {:.5f}, perplexity: {:.5f}".format(epoch, loss.item(), perplexity))
            total_loss = 0
            model.train()
    save_model(model, epoch)
    pickle.dump(perplexity_list, open('perplexity_list.p', 'wb'))
    pickle.dump(loss_list, open('loss_list.p', 'wb'))


plt.plot(loss_list)
plt.title('Loss')
plt.show()

plt.plot(perplexity_list)
plt.title('Perplexity')
plt.show()
