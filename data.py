from caption_model import *
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import spacy
from collections import Counter
from tqdm import tqdm
import json
import os
from PIL import Image
import matplotlib.pyplot as plt


class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self, pad_idx, batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs, targets


class Vocabulary:
    def __init__(self, freq_threshold):
        # setting the pre-reserved tokens int to string tokens
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

        # string to int tokens
        # its reverse dict self.itos
        self.stoi = {v: k for k, v in self.itos.items()}

        self.freq_threshold = freq_threshold
        self.spacy = spacy_eng = spacy.load("en_core_web_sm")

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text, spacy):
        return [token.text.lower() for token in spacy.tokenizer(text)]

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in tqdm(sentence_list):
            for word in self.tokenize(sentence, self.spacy):
                frequencies[word] += 1

                # add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text, self.spacy)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]


class FlickrDataset(Dataset):
    """
    FlickrDataset
    """
    def __init__(self, root_dir, captions_df, vocab, transform=None):
        self.root_dir = root_dir
        self.df = captions_df
        self.transform = transform

        # Get image and caption colum from the dataframe
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize vocabulary and build vocab
        self.vocab = vocab
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir, img_name)
        img = Image.open(img_location).convert("RGB")

        # apply the transfromation to the image
        if self.transform is not None:
            img = self.transform(img)

        # numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]

        return img, torch.tensor(caption_vec)


def show_image(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def build_vocab(captions_file_path):
    ## load all captions
    captions_df = pd.read_csv(captions_file_path)
    captions_list = captions_df['caption'].tolist()

    ## build vocabulary
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocab(captions_list)
    return vocab

def karpathy_split(captions_path, karpathy_json_path = 'dataset.json'):
    captions_df = pd.read_csv(captions_path)
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)
    train_images = []
    val_images = []
    test_images = []
    for img in data['images']:
        if img['split'] in {'train', 'restval'}:
                train_images.append(img['filename'])
        elif img['split'] in {'val'}:
                val_images.append(img['filename'])
        elif img['split'] in {'test'}:
                test_images.append(img['filename'])
    train_df = captions_df.loc[captions_df['image'].isin(train_images)].reset_index(drop=True)
    val_df = captions_df.loc[captions_df['image'].isin(val_images)].reset_index(drop=True)
    test_df = captions_df.loc[captions_df['image'].isin(test_images)].reset_index(drop=True)
    return train_df, val_df, test_df