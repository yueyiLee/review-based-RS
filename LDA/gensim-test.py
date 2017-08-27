import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities
from collections import defaultdict

# remove stop words and split each document, then we get texts
f = open('test-doc.txt')
stop_list = set('for a of the and to in'.split())
texts = [[word for word in line.strip().lower().split() if word not in stop_list] for line in f]
print('Text = ', texts)

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1] for text in texts]

# build the dictionary, a dict of (question, answer) pair
dictionary = corpora.Dictionary(texts)
print(dictionary.token2id)

# convert tokenized documents to vectors
new_doc = "Human computer tree"
# The function doc2bow() simply counts the number of occurrences of each distinct word,
# converts the word to its integer word id and returns the result as a sparse vector
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)
# transform the whole texts to vector(sparse)
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus)

# create a transformation, from initial model to tf-idf model
tfidf = models.TfidfModel(corpus)     # step 1 -- initialize a model
# "tfidf" is treated as a read-only object that can be used to convert any vector from the old representation
# (bag-of-words integer counts) to the new representation (TfIdf real-valued weights):
doc_bow = [(0, 1), (1, 1)]
print(tfidf[doc_bow])   # step 2 -- use the model to transform vectors
# apply a transformation to a whole corpus
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)  # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
lsi.print_topics(2)
for doc in corpus_lsi:
    print(doc)    # each line is associated with the probability for each topic

doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]  # convert the query to LSI space

# transform corpus to LSI space and index it for further calculating similarities
index = similarities.MatrixSimilarity(lsi[corpus])
sims = index[vec_lsi]  # perform a similarity query against the corpus
print(list(enumerate(sims)))  # print (document_number, document_similarity) 2-tuples
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims) # print sorted (document number, similarity score) 2-tuples



