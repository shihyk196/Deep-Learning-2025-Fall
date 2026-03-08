Lab03 Machine Translation

Dataset: English-Chinese Translation
Tatoeba – https://tatoeba.org/zh-cn/
XDailyDialog – https://github.com/liuzeming01/XDailyDialog
The dataset contains daily-style conversation sentence pairs
Select 50000 sentence pairs for training and validation
Select 200 sentence pairs for testing

In “Lab03.ipynb” and “network.py”
1.Build vanilla transformer (from Attention is All You Need) 
2.build the encoder, decoder 
3.Inside the encoder/decoder, build the attention mechanism (multi-head attention)
4.The parameter size of model should be less than 100M
5.Achieve at least 0.251-gram BLEU score on testing dataset
6.Achieve at least 0.12-gram BLEU score on testing dataset
