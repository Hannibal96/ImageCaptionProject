from DataLoader import *
from tqdm import tqdm

def save_model(model, num_epochs):
    path = "caption_model_E_"+str(num_epochs)+".torch"
    torch.save(model, path)


data_location = "."
BATCH_SIZE = 32
NUM_WORKER = 4

# defining the transform to be applied
transforms = T.Compose([
    T.Resize(226),
    T.RandomCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


dataset = FlickrDataset(root_dir=data_location+"/Images", captions_file=data_location+"/captions.txt",
                        transform=transforms)
print("Finished building the Dataset.")

# Hyperparams
embed_size = 300
vocab_size = len(dataset.vocab)
attention_dim = 256
encoder_dim = 2048
decoder_dim = 512
learning_rate = 3e-4


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#init model
model = EncoderDecoder(
    embed_size=300,
    vocab_size=len(dataset.vocab),
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 1
print_every = 100

pad_idx = dataset.vocab.stoi["<PAD>"]

data_loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=True,
    collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True)
)

loss_list = []
perplexity_list = []
total_loss = 0
for epoch in range(1, num_epochs + 1):
    for idx, (image, captions) in tqdm(enumerate(iter(data_loader))):
        image, captions = image.to(device), captions.to(device)

        # Zero the gradients.
        optimizer.zero_grad()

        # Feed forward
        outputs, attentions = model(image, captions)

        # Calculate the batch loss.
        targets = captions[:, 1:]
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
                caps, alphas = model.decoder.generate_caption(features, vocab=dataset.vocab)
                caption = ' '.join(caps)
                show_image(img[0], title=caption)
            print("Epoch: {} loss: {:.5f}, perplexity: {:.5f}".format(epoch, loss.item(), perplexity))
            total_loss = 0
            model.train()

    # save the latest model
    save_model(model, epoch)
    pickle.dump(loss_list, open("loss_list.p", "wb"))
    pickle.dump(perplexity_list, open("perplexity_list.p", "wb"))

plt.plot(loss_list)
plt.title('Loss')
plt.show()

plt.plot(perplexity_list)
plt.title('Perplexity')
plt.show()

