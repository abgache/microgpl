![ngpl_ico](ngpl.png)
# NanoGPL Beta 0.1
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red)](https://pytorch.org/)  
Small test generative pre-trained LAM (Linear Attention Mechanism).  
To change any setting, please go check ``config.json``.  

## What is LAM?
**LAM** stands for Linear Attention Mechanism, it's a [Transformer](https://arxiv.org/pdf/1706.03762)-like neural network architecture made for NGPL, it's way easier to understand, cheaper in resources but less efficient than the real Transformer arch even though they are both really similars.
### How does it works?
![shema](LAM%20-%20NanoGPL.png)  
> This shema show how it works from the input word list, to the tokens probabilities. Each bloc is explained down here.
<details>
<summary>1- Tokenizer</summary>
It's the first bloc, it's very useful for going from a string (e.g. "My name is abgache and i love machine learning!") to numbers (e.g. [45, 58, 563]).
It basically just create a token list (in this case it's ``model/tokenizer.json``but you can change it in the ``config.json`` file) and assign to each token an ID without any real meaning.
It uses the [BPE](https://huggingface.co/learn/llm-course/chapter6/5) (byte pair-encoding) algorithm to divide the stings into tokens.

To test it, use this command: 

```batch
   python main.py --tokenizer-test
```
</details>
<details>
<summary>2- Embeddings</summary>
It's the second bloc, it's very useful for giving a meaning to token IDs, it give a vector (In this case it has 256 dimensions, but you can change it in the ``config.json`` file) to each token to try to encode it's meaning.
The embeddings bloc outputs a list of vectors, each vector encodes the sense of a token.
At fist we generate them using a neural network, but then we save all tokens vectors in a list to save resources.

To test it, use this command: 

```batch
   python main.py --embedding-test
```
</details>
<details>
<summary>3- SPE</summary>
Soon...
</details>
<details>
<summary>4- Attention head(s)</summary>
Soon...
</details>
<details>
<summary>5- Feed forward neural network</summary>
It's the last bloc, it's a simple neural network that gets the input vector and try to predict the next token.
It's input layer is just 256 neurons (because in this case we have 256 dimensions in our vectors) and an output of the vocabulary size with a Softmax activation function.
</details>

---

## How to use it?
Before testing it, you will need to have a model, either you downloaded it or trained it by ourself.
### 1st stage : How to get NGPL files on my computer?
> [!IMPORTANT]
> To use NanoGPL, you will need: [Python 3.10](https://www.python.org/downloads/release/python-31010/) + [Git](https://git-scm.com/install/)  
**First, clone the repo:**  
```batch
   git clone https://github.com/abgache/NanoGPL.git
   cd NanoGPL
```
**Then, download the requirements:**  
```batch
   pip install -r requirements.txt
```

### 2nd stage : How to get a working model?
> [!NOTE]
> For the moment, the model is __not__ ready, soo theses command will, for most of them, probably not work. Please wait for the main release to use a stable version.  
**Option 1: download a model from huggingface/google drive**
To download the pre-trained models, please run this command:
```batch
   python main.py --download
```
> [!TIP]
> if you want the program to choose the best model for your hardware write ``n`` otherwise write ``y`` and choose one of the proposed models.

**Option 2: train your own model**
To train your own model, please run this command:
```batch
   python main.py --train
```
> [!TIP]
> The default dataset is Tiny Sheakspeare, it is not recommended for a real language model, it is only used for tests. I recommend downloading one from the internet, make sure it's in UTF-8 and change the path in ``config.json``. You can aswell run ``reset.bat`` to delete the saved tokenizer data, embedding tables, ...

---

### What are all the existing arguments?
- **--train :** Used to train a new model
- **--download :** Used to download one of the pretrained models
- **--predict :** Used to predict the next token of a string
- **--chat :** Used to use the model as a fully working chatbot
- **--tokenizer-test :** Used to test the tokenizer
- **--embedding-test :** Used to test the embedding model
- **--cpu :** Used to force the usage of the CPU
- **--cuda :** Used to force the usage of CUDA (NVIDIA Graphic cards)

---

> [!WARNING]
> I am __not__ responsible of any missusage for NanoGPL, any false information or anything he gave/told you. It is a test model, use it at your own risks.  
  
  
---

### TODO : 
- [X] Working tokenizer
- [X] Working embedding
- [X] Working SPE
- [ ] Working single attention head
- [ ] Working multiple attention heads
- [ ] Working and __trained__ final feed forward neural network
- [ ] Special file format for GPL models (maybe something like .gpl)
- [ ] Local API
- [ ] WebUI for ngpl
- [ ] Mobile version
- [ ] Comptable with AMD GPUs