from Train import *


# generate caption
def get_caps_from(features_tensors, model):
    # generate the caption
    model.eval()
    with torch.no_grad():
        features = model.encoder(features_tensors.to(device))
        caps, alphas = model.decoder.generate_caption(features, vocab=train_dataset.vocab)
        caption = ' '.join(caps)
        show_image(features_tensors[0], title=caption)

    return caps, alphas


# Show attention
def plot_attention(img, result, attention_plot):
    # untransform
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406

    img = img.numpy().transpose((1, 2, 0))
    temp_image = img

    fig = plt.figure(figsize=(15, 15))

    len_result = len(result)
    for ll in range(len_result):
        temp_att = attention_plot[ll].reshape(7, 7)

        ax = fig.add_subplot(len_result // 2, len_result // 2, ll + 1)
        ax.set_title(result[ll])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.7, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_location = '.'
    captions_file_path = "captions.txt"
    karpathy_json_path = 'Karpathy_data.json'

    transforms = T.Compose([T.Resize(226), T.RandomCrop(224), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    vocab = build_vocab(captions_file_path=captions_file_path)
    train, val, test = karpathy_split(captions_file_path, karpathy_json_path)
    train_dataset = FlickrDataset(root_dir='./Images', vocab=vocab, captions_df=train, transform=transforms)
    val_dataset = FlickrDataset(root_dir='./Images', vocab=vocab, captions_df=val, transform=transforms)
    test_dataset = FlickrDataset(root_dir='./Images', vocab=vocab, captions_df=test, transform=transforms)
    print("Finished building the Dataset.")


    pad_idx = vocab.stoi["<PAD>"]

    data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True,
        collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True)
    )

    # show any 1

    for i in range(1, 11):
        dataiter = iter(data_loader)
        images, _ = next(dataiter)

        model_name = './caption_model_E_'+str(i)+'.torch'
        model = torch.load(model_name)

        img = images[0].detach().clone()
        img1 = images[0].detach().clone()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        caps, alphas = get_caps_from(img.unsqueeze(0), model=model)

        plot_attention(img1, caps, alphas)
