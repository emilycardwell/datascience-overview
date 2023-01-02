# Deep NLPs

- Berkeley deep learning website

### RNNs

- y doesn’t have to be a part of the time series, you can use your time series to predict a different value
- get range of values (to make sure padded value isn’t in there)
    - df.max(axis=0) - maximum of each column
    - df.max(axis=1) - maximum of each row
    - df.min - minimum

# NLP

### Examples

- language model: attemprts to predict next word/character
- text classification: sentiment analysis
- sequence to sequence: translation

# Feed Recurrent NN with Words

- inputs → recurrent layer → outputs

### Clean the Data

- remove caps
- remove special characters and accents
- remove words that are:
    - too frequent
    - too rare

### Convert words to numbers

1. tokenization: each word = random int
    1. not great - it’s random
2. onehotencoder
    1. not great - too many corpuses (>10k)
3. high dimensional embedding:

## High dimensional embedding

- each word is represented by a vector
    - 4D embedding space : 4D latent space
- 1D embedding ≈ tokenization
    - order words according to negative vs positive (lose specific meanings)
- 2D embedding
    - add rank of relative abstraction
- **BEST: 30-300 dimensions**
    - **find one that’s specifically designed for the task**
    - find one that’s intrinsically good (less good)

### Arithmetic on words

- equal differences between vectors equal comparisons between words
- calculations between words

### Custom Embedding

- layers.Embedding
- X → tokenizer → embedding layer → recurrent layer → pad sequences → model
- NN learns the best representation of each word for your given task
- `X.shape = (n_sentences, max_sentence_length, embedding_dim)`

### Code it

```python
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

### Let's create some mock data
def get_mock_up_data():
    sentence_1 = 'Deep learning is super easy'
    sentence_2 = 'Deep learning was super bad and too long'
    sentence_3 = 'This is the best lecture of the camp!'

    X = [sentence_1, sentence_2, sentence_3]
    y = np.array([1., 0., 0.])

    ### Let's tokenize the vocabulary
    **tk = Tokenizer()
    tk.fit_on_texts(X)**
    **vocab_size = len(tk.word_index)**
    print(f'There are {vocab_size} different words in your corpus')
    **X_token = tk.texts_to_sequences(X)**

    ### Pad the inputs
    **X_pad = pad_sequences(X_token, dtype='float32', padding='post')**

    return X_pad, y, vocab_size

X_pad, y, vocab_size = get_mock_up_data()
print("X_pad.shape", X_pad.shape)
X_pad
```

`There are 16 different words in your corpus`

`X_pad.shape (3, 8)`

`array([[ 1.,  2.,  3.,  4.,  6.,  0.,  0.,  0.],`

       `[ 1.,  2.,  7.,  4.,  8.,  9., 10., 11.],`

       `[12.,  3.,  5., 13., 14., 15.,  5., 16.]], dtype=float32)`

```python
### Let's build the neural network now
from tensorflow.keras import layers, Sequential

# Size of your embedding space
embedding_size = 100

model = Sequential()
**model.add(layers.Embedding(
    input_dim=vocab_size+1,    # 16 +1 for the 0 padding (number of unique words)
    input_length=8,            # Max_sentence_length (optional, for model summary)
    output_dim=embedding_size, # size of vector rep each words
    mask_zero=True,            # Built-in masking layer**
))

model.add(layers.LSTM(20))
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()
```

`Total params: 11,401`

- number of params = (vocab_size + 1) * embedding_size

```python
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_pad, y, epochs=5, batch_size=16, verbose=0)
```

### Dealing with slow training

- You can reduce the size of your **embedding space** and **training data** to make training the model faster (esp. during prototyping).
- max_sentence_length:
    - 99 sentences of size 10
    - 1 sentence of size 1000 Padding will create a looooot of zeros in the padded tensor
    - pad your data to a reasonable size! And in general (sentiment analysis), you don't need the entire sentence to extract the information you need
- higher batch size
    - while prototyping: use 64, 128, 256
    - switch back for final runs

# Independent Embedding with Word2Vec

- input → wordvec to learn embedding → feed rnn → output
- “words to vectors”
- it automatically learns a representation - an embedding - for each word it was trained on

### Intuition

- split training sentences into sets of adjacent words

    1. One Hot encode all your words
    2. Train an auto-encoder to predict the word in the middle (e.g. brown) based on it's neighbors
    3. Use the latent space (middle layer) as embedding!

- input (text corpus), set params → output

## Implementation with Gensim

```python
### Let's get some text first

import tensorflow_datasets as tfds

train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"],
                                            batch_size=-1, as_supervised=True)

train_sentences, train_labels = tfds.as_numpy(train_data)
test_sentences, test_labels = tfds.as_numpy(test_data)

# Let's check two sentences
train_sentences[0:2]

# We have to convert the sentences into list of words! The computer won't do it for us
```

```python
# Let's convert the list of sentences to a list of lists of words with a Keras utility function

from tensorflow.keras.preprocessing.text import text_to_word_sequence

X_train = [text_to_word_sequence(_.decode("utf-8")) for _ in train_sentences]
X_test = [text_to_word_sequence(_.decode("utf-8")) for _ in test_sentences]

X_train[0:2]
```

```python
from gensim.models import Word2Vec

# This line trains an entire embedding for the words in your train set
word2vec = Word2Vec(sentences=X_train, vector_size=10)
```

```python
# Let's take a look at the representation of any word

word2vec.wv['hello']
```

`array([ 0.11349579, -0.1805743 ,  0.42932147, -0.29310516, -0.01752834,`

        `0.43636566,  0.27717137, -0.6297742 , -1.3054028 , -1.1144987 ]`)

```python
# Now let's look at the 10 closest words to `movie`

word2vec.wv.most_similar('movie', topn=10)
```

```python
# To control the size of the embedding space, you just have to set-up the `vector_size` keyword

word2vec = Word2Vec(sentences=X_train[:1000], vector_size=50) # We keep the training short by taking only 1000 sentences

len(word2vec.wv['computer'])
```

`50`

```python
# The Word2Vec learns a representation for words that are present more than `min_count` number of times
# This is to prevent learning representations based on a few examples only

word2vec = Word2Vec(sentences=X_train[:1000], vector_size=50, min_count=5)

try:
    len(word2vec.wv['columbian'])
except:
    print("word seen only less than 5 times, excluded from corpus")
```

`word seen only less than 5 times, excluded from corpus`

```python
# As mentioned ealier, Word2vec trains an internal Neural network whose goal is to predict a word in a corpus
# based on the words around it. This part of the sentence is called the window.
# Its size corresponds to the number of words around word W used to predict this word W

word2vec = Word2Vec(sentences=X_train[:10000], vector_size=100, window=5, min_count=1)
```

### Pre-trained Word2Vec : Transfer Learning

```python
# Instead of training it on your training set (especially if it is very small), you can directly
# load a pretrained embedding

import gensim.downloader

print(list(gensim.downloader.info()['models'].keys()))

model_wiki = gensim.downloader.load('glove-wiki-gigaword-50')
```

```python
model_wiki.most_similar('movie', topn=10)
```

`[('movies', 0.9322481155395508),`

 `('film', 0.9310100078582764),`

…

- words are better, but may not be as contextualized to your task

### Arithmetic on words

```python
word2vec = Word2Vec(sentences=X_train[:10000], vector_size=30, window=2, min_count=10)

v_queen = word2vec.wv['queen']
v_king = word2vec.wv['king']
v_man = word2vec.wv['man']

v_result = v_queen - v_king + v_man

word2vec.wv.similar_by_vector(v_result)

# You just did arithmetic directly on words!
```

`[('girl', 0.9156383275985718),`

 `('woman', 0.9044826030731201),`

…

## When to use what?

### Sequential(Layers.embedding + RNN)

- allows you to have a representation that is perfectly suited to your task! However, it increases the number of parameters to learn & increases:
    - the time of each epoch (more parameters to optimize during back-propagation)
    - the time to converge (because more parameters to find overall)

```python
# RNN
rnn = Sequential([
    layers.Embedding(input_dim=5000, input_length=20, output_dim=30, mask_zero=True),
    layers.LSTM(20),
    layers.Dense(1, activation="sigmoid")
])
```

`Total params: 154,101`

### Word2Vec, then RNN

- not specifically designed for your task (may be sub-optimal) but training it is very fast! You will also be able to optimize your RNN faster as you'll have less parameters
- Prefer Word2Vec on small corpus (esp. with transfer learning)

```python
word2vec = Word2Vec(sentences=X_train[:10000],
										vector_size=30, window=2, min_count=10)
```

### CNNs for NLP

- 2D convolutions on images
    - Each row corresponds to a coordinate in the embedding space.
    - This ordering is arbitrary.
    - We should never look at "half the embedding" of a word
    - Convolutions should not look for "relations" between the top rows and bottom rows!
- 1D convolutions
    - convolutions that "slide" along the word axis, word-by-word
    - includes the whole vector of each word
    - **Kernel-size** (here 2) equals the **number of words upon which the convolution is computed** (through the kernel weights)
        - captures features in their context

```python
# Conv1D
cnn = Sequential([
    layers.Embedding(input_dim=5000, input_length=20, output_dim=30, mask_zero=True),
    **layers.Conv1D(20, kernel_size=3),**
    layers.Flatten(),
    layers.Dense(1, activation="sigmoid"),
])
```

`Total params: 152,181`

# Challenges

### 1

```python
import pandas as pd
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Sequential
from tensorflow.keras import callbacks
```

```python
def load_data(percentage_of_sentences=None):
    train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"], batch_size=-1, as_supervised=True)

    train_sentences, y_train = tfds.as_numpy(train_data)
    test_sentences, y_test = tfds.as_numpy(test_data)

    # Take only a given percentage of the entire data
    if percentage_of_sentences is not None:
        assert(percentage_of_sentences> 0 and percentage_of_sentences<=100)

        len_train = int(percentage_of_sentences/100*len(train_sentences))
        train_sentences, y_train = train_sentences[:len_train], y_train[:len_train]

        len_test = int(percentage_of_sentences/100*len(test_sentences))
        test_sentences, y_test = test_sentences[:len_test], y_test[:len_test]

    X_train = [text_to_word_sequence(_.decode("utf-8")) for _ in train_sentences]
    X_test = [text_to_word_sequence(_.decode("utf-8")) for _ in test_sentences]

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data(percentage_of_sentences=10)
```

```python
X_tr_df = pd.DataFrame(X_train)
X_tr_df.shape
```

```python
# This initializes a Keras utilities that does all the tokenization for you
tokenizer = Tokenizer()

# The tokenization learns a dictionary that maps a token (integer) to each word
# It can be done only on the train set - we are not supposed to know the test set!
# This tokenization also lowercases your words, apply some filters, and so on - you can check the doc if you want
tokenizer.fit_on_texts(X_train)

# We apply the tokenization to the train and test set
X_train_token = tokenizer.texts_to_sequences(X_train)
X_test_token = tokenizer.texts_to_sequences(X_test)
```

```python
vocab_size = len(tokenizer.word_index)
```

```python
X_train_pad = pad_sequences(X_train_token, dtype='float64', padding='post')
X_test_pad = pad_sequences(X_test_token, dtype='float64', padding='post')
```

```python
embedding_size = 1

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size+1,
                           output_dim=embedding_size,
                           mask_zero=True))
model.add(layers.GRU(10))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
```

```python
es = callbacks.EarlyStopping(patience=4)

model.fit(X_train_pad, y_train, validation_split=0.3,
          epochs = 10, batch_size=32,
          callbacks=[es])
```

---

```python
X_train_pad = pad_sequences(X_train_token, dtype='float64', padding='post', maxlen=200)
X_test_pad = pad_sequences(X_test_token, dtype='float64', padding='post', maxlen=200)
```

```python
embedding_size = 1

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size+1,
                           output_dim=embedding_size,
                           mask_zero=True))
model.add(layers.GRU(10))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
```

```python
es = callbacks.EarlyStopping(patience=4)

model.fit(X_train_pad, y_train, validation_split=0.3,
          epochs = 10, batch_size=32,
          callbacks=[es])
```

### 2

```python
from gensim.models import Word2Vec

word2vec = Word2Vec(sentences=X_train)
wv = word2vec.wv
len(wv['hello'])
```

`100`

```python
embedding_size = 100
```

```python
word2vec_5 = Word2Vec(sentences=X_train,
                      vector_size=30,
                      min_count=3,
                      window=2)
wv5 = word2vec_5.wv
```

```python
def embed_sentence(word2vec, sentence):

    wv = word2vec.wv

    embedded_sentence = []

    for word in sentence:
        if word in wv:
            embed_word = wv[word]
            embedded_sentence.append(embed_word)
        else:
            pass

    return np.array(embedded_sentence)

embedded_sentence = embed_sentence(word2vec, example)
```

```python
def embedding(word2vec, sentences):

    embedded_sentences = []

    for sentence in sentences:
        embedded_sentences.append(embed_sentence(word2vec, sentence))

    return np.array(embedded_sentences)

X_embed_train = embedding(word2vec, X_train)
X_embed_test = embedding(word2vec, X_test)
```

```python
X_train_pad = pad_sequences(X_embed_train, dtype='float64', padding='post', maxlen=64)
X_test_pad = pad_sequences(X_embed_test, dtype='float64', padding='post', maxlen=64)
X_train_pad.shape
```

`(2500, 64, 100)`

### 3

```python
def embed_sentence(word2vec, sentence):
    embedded_sentence = []
    for word in sentence:
        if word in word2vec.wv:
            embedded_sentence.append(word2vec.wv[word])

    return np.array(embedded_sentence)

# Function that converts a list of sentences into a list of matrices
def embedding(word2vec, sentences):
    embed = []

    for sentence in sentences:
        embedded_sentence = embed_sentence(word2vec, sentence)
        embed.append(embedded_sentence)

    return embed

# Embed the training and test sentences
X_train_embed = embedding(word2vec, X_train)
X_test_embed = embedding(word2vec, X_test)

# Pad the training and test embedded sentences
X_train_pad = pad_sequences(X_train_embed, dtype='float32', padding='post', maxlen=200)
X_test_pad = pad_sequences(X_test_embed, dtype='float32', padding='post', maxlen=200)
```

```python
baseline_accuracy = np.unique(y_train, return_counts=True)[1][0] / np.sum(np.unique(y_train, return_counts=True)[1])
```

```python
def init_model(input_shape):
    model = models.Sequential()
    model.add(layers.Masking(mask_value=-1000, input_shape=input_shape))
    model.add(layers.LSTM(20))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    return model
```

```python
def fit_mod(model, X, y):
    es = EarlyStopping(patience=3)

    model.fit(X, y, validation_split=0.3,
            batch_size=64, epochs=10, callbacks=[es])
    return model
```

```python
model.evaluate(X_test_pad, y_test)
```

```python
import gensim.downloader

word2vec_transfer = gensim.downloader.load('glove-wiki-gigaword-50')
```

```python
def embed_sentence_with_TF(word2vec, sentence):
    embedded_sentence = []
    for word in sentence:
        if word in word2vec:
            embedded_sentence.append(word2vec[word])

    return np.array(embedded_sentence)

# Function that converts a list of sentences into a list of matrices
def embedding(word2vec, sentences):
    embed = []

    for sentence in sentences:
        embedded_sentence = embed_sentence_with_TF(word2vec, sentence)
        embed.append(embedded_sentence)

    return embed

# Embed the training and test sentences
X_train_embed_2 = embedding(word2vec_transfer, X_train)
X_test_embed_2 = embedding(word2vec_transfer, X_test)
```

```python
X_train_pad_2 = pad_sequences(X_train_embed_2, dtype='float32', padding='post', maxlen=64)
X_test_pad_2 = pad_sequences(X_test_embed_2, dtype='float32', padding='post', maxlen=64)
X_train_pad_2.shape
```

`(2500, 64, 50)`

```python
model = init_model((64, 50))
model = fit_mod(model, X_train_pad_2, y_train)
```

```python
res = model.evaluate(X_test_pad_2, y_test)
```

### 4

```python
def tokenize(X):
    tk = Tokenizer()
    tk.fit_on_texts(X)
    vocab_size = len(tk.word_index)
    X_token = tk.texts_to_sequences(X)
    X_pad = pad_sequences(X_token, dtype='float32', padding='post', maxlen=150)
    return X_pad, vocab_size
```

```python
X_tr_pad, vocab_size = tokenize(X_train)
vocab_size = vocab_size + 1
input_len = X_tr_pad.shape[1]
```

```python
X_tst_pad, _ = tokenize(X_test)
```

```python
embed_size = 100

def model_inst():
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size,
                               input_length=input_len,
                               output_dim=embed_size,
                               mask_zero=True))
    model.add(layers.Conv1D(10, kernel_size=3))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model
```

```python
model_cnn = model_inst()
model_cnn.summary()
```

```python
def model_fit(model, X_train, y_train):
    model.fit(X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.3,
            callbacks=[es]
            )
    return model

model_cnn = model_fit(model_cnn, X_tr_pad, y_train)

res = model_cnn.evaluate(X_tst_pad, y_test, verbose=0)

```

---

```python
import gensim.downloader

word2vec_transfer = gensim.downloader.load('glove-wiki-gigaword-50')
```

```python
def embed_sentence_with_TF(word2vec, sentence):
    embedded_sentence = []
    for word in sentence:
        if word in word2vec:
            embedded_sentence.append(word2vec[word])

    return np.array(embedded_sentence)

# Function that converts a list of sentences into a list of matrices
def embedding(word2vec, sentences):
    embed = []

    for sentence in sentences:
        embedded_sentence = embed_sentence_with_TF(word2vec, sentence)
        embed.append(embedded_sentence)

    return embed

# Embed the training and test sentences
X_train_embed = embedding(word2vec_transfer, X_train)
X_test_embed = embedding(word2vec_transfer, X_test)
```

```python
X_train_pad = pad_sequences(X_train_embed, dtype='float32', padding='post', maxlen=64)
X_test_pad = pad_sequences(X_test_embed, dtype='float32', padding='post', maxlen=64)
```

```python
X_train_pad.shape
```

```python
def model_inst_1(input_shape):
    model = Sequential()
    model.add(layers.Masking(mask_value=-1000, input_shape=input_shape))
    model.add(layers.LSTM(20))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    return model
```

```python
model = model_inst_1((64,50))
model = model_fit(model, X_train_pad, y_train)
model.evaluate(X_test_pad, y_test)
```

### 5

```python
from tensorflow.keras.datasets import imdb

def load_data(percentage_of_sentences=None):
    # Load the data
    (sentences_train, y_train), (sentences_test, y_test) = imdb.load_data()

    # Take only a given percentage of the entire data
    if percentage_of_sentences is not None:
        assert(percentage_of_sentences> 0 and percentage_of_sentences<=100)

        len_train = int(percentage_of_sentences/100*len(sentences_train))
        sentences_train = sentences_train[:len_train]
        y_train = y_train[:len_train]

        len_test = int(percentage_of_sentences/100*len(sentences_test))
        sentences_test = sentences_test[:len_test]
        y_test = y_test[:len_test]

    # Load the {interger: word} representation
    word_to_id = imdb.get_word_index()
    word_to_id = {k:(v+3) for k,v in word_to_id.items()}
    for i, w in enumerate(['<PAD>', '<START>', '<UNK>', '<UNUSED>']):
        word_to_id[w] = i

    id_to_word = {v:k for k, v in word_to_id.items()}

    # Convert the list of integers to list of words (str)
    X_train = [' '.join([id_to_word[_] for _ in sentence[1:]]) for sentence in sentences_train]

    return X_train

### Just run this cell to load the data
X = load_data(percentage_of_sentences=10)
```

```python
def x_y_split(string):
    if len(string) >= 300:
        return (string[:300], string[300])
    else:
        return None
```

```python
def get_all_splits(data):
    string_set = []
    y_set = []
    for d in data:
        if len(d) > 300:
            string, y = x_y_split(d)
            string_set.append(string)
            y_set.append(y)
        else:
            pass
    return string_set, y_set
```

```python
strings, y = get_all_splits(X)
string_train, string_test, y_train, y_test = train_test_split(strings, y,
																															test_size=0.3)
```

# Recap

```python
mae_baseline = np.mean(np.abs(y - y_mean))
```

---

```python
import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
```

---

```python
def build_model_nlp():
    model = Sequential([
        layers.Embedding(input_dim=vocab_size+1, input_length=maxlen,
                         output_dim=embedding_size, mask_zero=True),
        layers.Conv1D(10, kernel_size=15, padding='same', activation="relu"),
        layers.Conv1D(10, kernel_size=10, padding='same', activation="relu"),
        layers.Flatten(),
        layers.Dense(30, activation='relu'),
        layers.Dropout(0.15),
        layers.Dense(1, activation='relu'),
    ])

    model.compile(loss="mse", optimizer=Adam(learning_rate=1e-4), metrics=['mae'])
    return model

model_nlp = build_model_nlp()
```

```python
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard

es = EarlyStopping(patience=2)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model_nlp = build_model_nlp()
model_nlp.fit(X_pad, y,
          validation_split=0.3,
          epochs=50,
          batch_size=32,
          callbacks=[es, tensorboard_callback]
          )
```

```python
%tensorboard --logdir logs/fit
```

---

### Combined RNN + Tabular data inputs

```python
#Define Inputs and Outputs of NLP model as with Numeric Model

model_nlp = build_model_nlp() # comment-out to keep pre-trained weights not to start from scratch
input_text = model_nlp.input
output_text = model_nlp.output

model_num = build_model_num() # comment-out to keep pre-trained weights not to start from scratch
input_num = model_num.input
output_num = model_num.output
```

```python
inputs = **[input_text, input_num]**

combined = layers.concatenate(**[output_text, output_num]**)

x = layers.Dense(10, activation="relu")(combined)

outputs = layers.Dense(1, activation="linear")(x)

model_combined = models.Model**(inputs=inputs, outputs=outputs)**
```

```python
import tensorflow as tf
tf.keras.utils.plot_model(model_combined, "multi_input_model.png",
													show_shapes=True)
```

```python
model_combined.compile(loss="mse",
											 optimizer=Adam(learning_rate=1e-4),
											 metrics=['mae'])

es = EarlyStopping(patience=2)

model_combined.fit(x=[X_pad, X_num],
                   y=y,
                   validation_split=0.3,
                   epochs=100,
                   batch_size=32,
                   callbacks=[es])
```
