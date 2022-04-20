# Text Generation with LSTMs in PyTorch

This repository presents a model for text generation using LSTM . 


## 1. Files
- The ``data`` directory contains the text which we will work with. 
- The ``src`` directory contains the file ``model.py``which contains the neural net definition
- The ``utils`` directory contains helper functions such as the preprocessor and the parser
- The ``weights`` directory contains the trained weights.

The best results were obtained by training the model with the following parameters:
```
python -B main.py 
```
The weights of the trained model are stored in the ``weights/``directory. 
To generate text, we are going to load the weights of the trained model as follows:
```
python -B main.py --load_model True --model [weights/your_model.pt]
```


