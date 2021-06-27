# ImageCaptionProject

## Introduction:
Image captioning is a task for generating textual description of an image. for example:
<p align="center">
  <img src="./imgs_for_readme/examples.png" width="600"/>
</p>

### Previous work:  
There is a lot of approaches and trials to approach the task, the most celebrated ones are referred to bellow, most of the work done in the last years is a combination of those works.
1.	Andrej Karpathy Ph.D. work was done about image description:
[***Deep Visual-Semantic Alignments for Generating Image Descriptions***](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)
the work mostly showed the idea of RCNN, scoring method and the idea to build the sentence by looking at different parts of the image.
2.	Google builds on the previous work using LSTM instead of plain RNN and beam search:
3.	Microsoft architecture added the ability to identify landmarks and celebrities:
[***Rich Image Captioning in the Wild***](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/ImageCaptionInWild-1.pdf)

### Scope
All The above models are based upon an architicture which consists of a CNN that learns a feature-map followed by an RNN that generates the caption:
<p align="center">
  <img src="./imgs_for_readme/Arch_init.png" width="400"/>
</p>
In this project we will implement a model which is based on: 

[***Show, Attend and Tell***](https://arxiv.org/pdf/1502.03044.pdf)
where soft attention mechanism has been added to the architicture, which at each time-step the models learns *where to look*. 
<p align="center">
  <img src="./imgs_for_readme/dogtie.png" width="800"/>
</p>



## Overview:

### Model
Our model is a decoder-encoder which have 4 main parts: 
1. A convolutional neural network that extracts features
from the images (Encoder) - for this part we decided to use a pretrained **Resnet-50**
2. an attention mechanism that weights the
image features - 
3. Embeddings for the tokens in the captions -  [*Pre-trained word embeddings*](https://pytorch.org/text/stable/vocab.html#glove) added as optional
4. An RNN that generates captions to
describe the weighted image features (Decoder) - We will use an LSTM 

<p align="center">
  <img src="./imgs_for_readme/Overall_arch.png" width="800"/>
</p>


### Data
We will use the **Flicker8k** dataset which can be downloaded from [***here***](https://www.kaggle.com/adityajn105/flickr8k)
that have 8092 images along with 5 _reference_ captions per image (used for [evaluation](https://github.com/Hannibal96/ImageCaptionProject/blob/master/README.md#evaluation-metric)

#### training, validation and test split:
- We will use the popular Karpathy split, which can be downloaded from [***here***](https://cs.stanford.edu/people/karpathy/deepimagesent/)

### Evaluation Metric
By far the most popular metric for evaluating generated sentences is the [***Blue score***](https://www.aclweb.org/anthology/P02-1040.pdf)
Which uses the 5 reference captions provided in the dataset to evaluate the generated caption.

## Usage 

### Dependencies
- software dependencies can be found in [requirements.txt]https://github.com/Hannibal96/ImageCaptionProject/blob/master/requirements.txt)
- Download the dataset as described above
- The dataset contains :
    - Images folder which contains all the images 
    - captions.txt which contains 5 different captions for each image(name) in Images folder
- karpathy split is availabe in [Karpathy_split.json]https://github.com/Hannibal96/ImageCaptionProject/blob/master/Karpathy_split.json)

### Data Pipeline
#### Images
Since we are using a pretrained CNN, we need to preprocess the images to have similar shape/stats that this model expect.
[This page](https://pytorch.org/vision/stable/models.html) contains information regarding the preprocessing needed.
in our case, we are using the following preprocessing:
```python
transforms = T.Compose([
    T.Resize(226),
    T.RandomCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
```
### Captions
See [data.py]https://github.com/Hannibal96/ImageCaptionProject/blob/master/data.py) for details and implementation.
- We start by building the vocabulary from the *full* data with the addition of the special tokens:`<PAD>,<SOS>,<EOS>,<UNK>` ;see build_vocab()
- for each caption we add the `<SOS>,<EOS>` to the beginning/end respectively, replace rare words with `<UNK>` 
- Since we are passing the captions as tensors, we need to maintain fixed size inputs. therefore we add `<PAD>` tokens at the end of shorter captions.
- using the vocabulary, we numericalize the captions using the `stoi` to be compatible with PyTorch Embedding layer.
