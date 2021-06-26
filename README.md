# ImageCaptionProject

## Introduction:
Image captioning is a task for generating textual description of an image. 
![](./imgs_for_readme/examples.png)

### Previous work:  
There is a lot of approaches and trials to approach the task, the most celebrated ones are referred to bellow, most of the work done in the last years is a combination of those works.
1.	Andrej Karpathy Ph.D. work was done about image description:
https://cs.stanford.edu/people/karpathy/cvpr2015.pdf
the work mostly showed the idea of RCNN, scoring method and the idea to build the sentence by looking at different parts of the image.
2.	Google builds on the previous work using LSTM instead of plain RNN and beam search:
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/ImageCaptionInWild-1.pdf
3.	Microsoft architecture added the ability to identify landmarks and celebrities:
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/ImageCaptionInWild-1.pdf


## Overview:
The model make use in the idea of *transfer learning*, *encoder-decoder architecture* and *attention*.
The forward action of the model consists of two parts, Encoder and Decoder.
The Encoder is a CNN that takes image as an input and extract its features. The features will be the last layer before the final output layer.
For purposes of efficiency, we will use pretrained model of resnet50 and use its output features as the input for the decoder. 

While the Decoder part will be LSTM instead of plain RNN layer.
The output of the decoder then goes into the Decoder that in addition receive the sentence that describing the photo in case it is the training part or the generated sentence so far in case of prediction.

