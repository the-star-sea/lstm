# Text Generation with Bi-LSTMs in PyTorch

This repository presents a model for text generation using Bi-LSTM and attention neural networks. 


## 1. Files
- The ``data`` directory contains the text which we will work with. 
- The ``src`` directory contains the file ``model.py``which contains the neural net definition
- The ``utils`` directory contains helper functions such as the preprocessor and the parser
- The ``weights`` directory contains the trained weights.

## 2. The model
The architecture of the proposed neural network consists of an embedding layer followed by a Bi-LSTM as well as attention layer. Right after, the latter LSTM is connected to a linear layer. The following image describes the model architecure. 



## 3. Demo
For this demos we are going to make use of the book that is in the ``data/book`` directory, the credentials of the book are:
```
book_name: 诡秘之主

```
First lines of the book:
```
The train rushed down the hill, with a long shrieking whistle, and then
began to go more and more slowly. Thomas had brushed Jack off and
thanked him for the coin that he put in his hand, and with the bag in
one hand and the stool in the other now went out onto the platform and
down the steps, Jack closely following.
```
The best results were obtained by training the model with the following parameters:
```
python -B main.py 
```
The weights of the trained model are stored in the ``weights/``directory. 
To generate text, we are going to load the weights of the trained model as follows:
```
python -B main.py --load_model True --model [weights/your_model.pt]
```

```

