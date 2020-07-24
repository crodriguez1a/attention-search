# [wip] attention-search

A minimal vector search implementation leveraging **Scaled Dot-Product Attention** as defined in [Attention is All You Need](https://arxiv.org/abs/1706.03762):

> "We call our particular attention “Scaled Dot-Product Attention”. The input consists of queries and keys of dimension d<sub>k</sub>, and values of dimension d<sub>v</sub>
. We compute the dot products of the query with all keys, divide each by the square root of d<sub>k</sub>, and apply a softmax function to obtain the weights on the values."

Applying the attention function in this manner ultimately allows for rapid and simultaneous [dot product] scoring of N number of possible search results.

#### Examples:

```
# search index
values = ["apples", "cookies", "oranges", "grapes"]

# vector representation of search index
emb = np.concatenate([conv_vec(i) for i in values])

# search query
query = conv_vec("mandarin")

attention_search(query, emb, values, n_results=2)
# => ['oranges', 'apples']


```

[See Notebook](notebooks/)

---
**References**:

- https://medium.com/analytics-vidhya/masking-in-transformers-self-attention-mechanism-bad3c9ec235c
- http://jalammar.github.io/illustrated-transformer/
- http://nlp.seas.harvard.edu/2018/04/03/attention.htmltte
- https://towardsdatascience.com/learning-attention-mechanism-from-scratch-f08706aaf6b6
- https://developers.google.com/machine-learning/clustering/similarity/measuring-similarity
