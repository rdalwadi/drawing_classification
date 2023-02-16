import numpy as np
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as normalize_func
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import spacy
#import tensorflow_hub as hub
#import tensorflow_text
#import tensorflow as tf


class GoogleALBERTVectorizer(object):
    # https://tfhub.dev/tensorflow/albert_en_base/3
    def __init__(self):
        self.preprocessor = hub.KerasLayer("http://tfhub.dev/tensorflow/albert_en_preprocess/3")
        self.encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/3",trainable=True)
    
    def fit(self, documents, targets=None):
        return self
   
    def transform(self, documents):
        encoding = self.encoder(self.preprocessor(documents))["pooled_output"]
        return encoding

    def fit_transform(self, documents, targets=None):
        return self.fit(documents, targets).transform(documents)


class GoogleBERTVectorizer(object):
    # https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1
    def __init__(self):
        self.preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        self.encoder = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1")
    
    def fit(self, documents, targets=None):
        return self
   
    def transform(self, documents):
        encoding = self.encoder(self.preprocessor(documents))["default"]
        return encoding
    
    def fit_transform(self, documents, targets=None):
        return self.fit(documents, targets).transform(documents)


    
class GoogleFeedForwardNeuralNetLanguageModelVectorizer(object):
    # https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2
    def __init__(self):
        self.embedder = hub.load("https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2")
    
    def fit(self, documents, targets=None):
        return self
   
    def transform(self, documents):
        encoding = self.embedder(documents)
        return encoding
    
    def fit_transform(self, documents, targets=None):
        return self.fit(documents, targets).transform(documents)


class GoogleUniversalSentenceEncoderVectorizer(object):
    # https://tfhub.dev/google/universal-sentence-encoder-large/5
    def __init__(self,  model='large'):
        if  model == 'large':
            self.embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        else:
            self.embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    def fit(self, documents, targets=None):
        return self
   
    def transform(self, documents):
        encoding = self.embedder(documents)
        return encoding
    
    def fit_transform(self, documents, targets=None):
        return self.fit(documents, targets).transform(documents)


class SpaCyVectorizer(object):
    # https://spacy.io
    def __init__(self, n_components=200, normalize=True):
        self.n_components = n_components
        self.normalize = normalize
        self.svd = TruncatedSVD(n_components=self.n_components)
        self.n_components_embedding = 300
        self.nlp = spacy.load("en_core_web_lg")

    def fit(self, documents, targets=None):
        encoding = np.vstack([self._transform(document) for document in documents])
        self.svd = self.svd.fit(encoding)
        return self
    
    def _transform(self, document):
        tokens = self.nlp(document)
        embedding_list = [token.vector for token in tokens if token.has_vector]
        if len(embedding_list) > 0:
            vec = np.vstack(embedding_list).sum(axis=0)
        else:
            vec = np.zeros(self.n_components_embedding)
        return vec
        
    def transform(self, documents):
        encoding = np.vstack([self._transform(document) for document in documents])
        encoding = self.svd.transform(encoding)
        if self.normalize:
            encoding = normalize_func(encoding)
        return encoding
    
    def fit_transform(self, documents, targets=None):
        return self.fit(documents, targets).transform(documents)

    
class SpaCySVOVectorizer(object):
    # https://spacy.io
    def __init__(self, split_sentences=False, n_components=200, normalize=True):
        self.n_components = n_components
        self.normalize = normalize
        self.svd = TruncatedSVD(n_components=self.n_components)
        self.split_sentences = split_sentences
        self.n_components_embedding = 300
        self.nlp = spacy.load("en_core_web_lg")

    def fit(self, documents, targets=None):
        encoding = np.vstack([self._transform(document) for document in documents])
        self.svd = self.svd.fit(encoding)
        return self
    
    def _extract_svo(self, doc):
        # based on https://github.com/Dimev/Spacy-SVO-extraction
        # object and subject constants
        SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent", "expl"}
        OBJECT_DEPS = {"dobj", "dative", "attr", "oprd"}
        sub = []
        ve = []
        obj = []
        for token in doc:
            # is this a subject?
            if token.pos_ == "NOUN" and token.dep_ in SUBJECT_DEPS:
                sub.append(token.text)
            # is this a verb?
            if token.pos_ == "VERB":
                ve.append(token.text)
            # is this an object?
            if token.pos_ == "NOUN" and token.dep_ in OBJECT_DEPS:
                obj.append(token.text)
        return " ".join(sub).strip().lower(), " ".join(ve).strip().lower(), " ".join(obj).strip().lower()

    def _doc_svo2vec(self, doc_svo):
        vecs = []
        for part in doc_svo:
            tokens = self.nlp(part)
            embedding_list = [token.vector for token in tokens if token.has_vector]
            if len(embedding_list) > 0:
                vec = np.vstack(embedding_list).sum(axis=0)
            else:
                vec = np.zeros(self.n_components_embedding)
            vecs.append(vec)
        return np.hstack(vecs)

    def _transform(self, document):
        doc = self.nlp(document)
        if self.split_sentences is True:
            sentences = [sen for sen in doc.sents]
            docs_svo = [self._extract_svo(sentence) for sentence in sentences]
        else:
            docs_svo = [self._extract_svo(doc)]
        embedding_list = [self._doc_svo2vec(doc_svo) for doc_svo in docs_svo]
        if len(embedding_list) > 0:
            vec = np.vstack(embedding_list).sum(axis=0)
        else:
            vec = np.zeros(self.n_components_embedding*3)
        return vec
        
    def transform(self, documents):
        encoding = np.vstack([self._transform(document) for document in documents])
        encoding = self.svd.transform(encoding)
        if self.normalize:
            encoding = normalize_func(encoding)
        return encoding
    
    def fit_transform(self, documents, targets=None):
        return self.fit(documents, targets).transform(documents)


def glove_embeddings_dict_init(dir_name, model, n_components_embedding):
    if model == 'large':
        assert n_components_embedding in [300]
        fname = '%s/glove.840B.%dd.txt'%(dir_name, n_components_embedding)
    elif model == 'medium':
        assert n_components_embedding in [300]
        fname = '%s/glove.42B.%dd.txt'%(dir_name, n_components_embedding)
    elif model == 'small':
        assert n_components_embedding in [50,100,200,300]
        fname = '%s/glove.6B.%dd.txt'%(dir_name, n_components_embedding)
    else:
        assert False, 'unknown model:%s'%model

    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    embeddings_dict = dict()
    with open(fname) as f:
        for line in f:
            values = line.split()
            word = values[0]
            # check that next values can be parsed as a vector
            if isfloat(values[1]):
                vector = np.asarray(values[1:], "float32")
                # check that the vector size is consistent
                if len(vector) == n_components_embedding:
                    embeddings_dict[word] = vector
    return embeddings_dict

class GloVeVectorizer(object):
    def __init__(self, n_components_embedding=300, normalize=True, ngram=1, dir_name='glove', model='large', embeddings_dict=None):
        self.n_components_embedding = n_components_embedding
        self.normalize = normalize
        self.ngram = ngram
        if embeddings_dict is None:
            self.embeddings_dict = glove_embeddings_dict_init(dir_name, model, n_components_embedding)
        else:
            self.embeddings_dict = embeddings_dict

    def fit(self, documents, targets=None):
        return self
    
    def _transform(self, document):
        unknown_vec = np.zeros(self.n_components_embedding*self.ngram)
        if len(document) == 0:
            return unknown_vec 
        words = document.lower().split()
        vec = [self.embeddings_dict[word].reshape(1,-1) for word in words if word in self.embeddings_dict]
        if self.ngram > 1:
            # concatenate a number=ngram of successive vectors
            higher_ngram_vec = []
            n = len(vec)
            for i in range(n-self.ngram):
                vecs = vec[i:i+self.ngram]
                vecs = np.vstack(vecs).reshape(-1)
                higher_ngram_vec.append(vecs)
            encoding = np.vstack(higher_ngram_vec)
        else:
            encoding = np.vstack(vec)
        encoding = encoding.sum(axis=0).reshape(-1)
        if self.normalize:
            encoding = normalize_func(encoding.reshape(1, -1)).reshape(-1)
        return encoding

    def transform(self, documents):
        encoding = np.vstack([self._transform(document) for document in documents])
        return encoding
    
    def fit_transform(self, documents, targets=None):
        return self.fit(documents, targets).transform(documents)
    
    def find_closest_embeddings(self, embedding, n_neighbors=5):
        return sorted(self.embeddings_dict.keys(), key=lambda word: sp.spatial.distance.euclidean(self.embeddings_dict[word], embedding))[:n_neighbors]

    def find_closest_words(self, word, n_neighbors=5):
        embedding = self.embeddings_dict[word]
        return self.find_closest_embeddings(embedding, n_neighbors)

    def find_equivalent_words(self, word1, is_to_word2, as_word3, is_to_n_words=5):
        A = embedding = self.embeddings_dict[word1]
        B = embedding = self.embeddings_dict[is_to_word2]
        C = embedding = self.embeddings_dict[as_word3]
        D = B-A+C
        return self.find_closest_embeddings(D, is_to_n_words)


class TfIdfVectorizer(object):
    def __init__(self, n_components=None, normalize=True, stop_words='english', analyzer='word', ngram_range=(1,1)):
        self.n_components = n_components
        self.normalize = normalize
        if analyzer != 'word':
            stop_words = None
        self.vectorizer = TfidfVectorizer(stop_words=stop_words, analyzer=analyzer, ngram_range=ngram_range)
        if self.n_components is not None:
            self.svd = TruncatedSVD(n_components=self.n_components)
        else:
            self.svd = None

    def _make_corpus(self, documents, targets):
        # concatenate all strings belonging to the same class in a single document
        corpus = []
        for reference_target in sorted(set(targets)):
            documents_target_i = [document for document, target in zip(documents, targets) if target == reference_target]
            corpus.append(' '.join(documents_target_i))
        return corpus

    def fit(self, documents, targets=None):
        corpus = self._make_corpus(documents, targets)
        self.vectorizer = self.vectorizer.fit(corpus)
        if self.svd is not None:
            X = self.vectorizer.transform(documents)
            self.svd = self.svd.fit(X)
        return self
    
    def transform(self, documents):
        X = self.vectorizer.transform(documents)
        if self.svd is not None:
            X = self.svd.transform(X)
        if self.normalize:
            X = normalize_func(X)
        return X
    
    def fit_transform(self, documents, targets=None):
        return self.fit(documents, targets).transform(documents)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names() 

def text_to_tfidf(doc, vectorizer):
    encoding = vectorizer.transform([doc])
    words = vectorizer.get_feature_names() 
    map_word2tfidf = {words[i]:encoding[0,i] for i in encoding.indices}
    tfidf_vec = np.array([map_word2tfidf[w] if w in map_word2tfidf else 0 for w in doc.lower().split()])
    return tfidf_vec

class TfIdfGloVeVectorizer(object):
    def __init__(self, n_components=100, n_components_embedding=300, normalize=True, stop_words='english', dir_name='glove', model='large', embeddings_dict=None):
        self.n_components = n_components
        self.n_components_embedding = n_components_embedding
        self.normalize = normalize
        if embeddings_dict is None:
            self.embeddings_dict = glove_embeddings_dict_init(dir_name, model, n_components_embedding)
        else:
            self.embeddings_dict = embeddings_dict
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.svd = TruncatedSVD(n_components=self.n_components)

    def _make_corpus(self, documents, targets):
        # concatenate all strings belonging to the same class in a single document
        corpus = []
        for reference_target in sorted(set(targets)):
            documents_target_i = [document for document, target in zip(documents, targets) if target == reference_target]
            corpus.append(' '.join(documents_target_i))
        return corpus

    def fit(self, documents, targets=None):
        corpus = self._make_corpus(documents, targets)
        self.vectorizer = self.vectorizer.fit(corpus)
        encoding = np.vstack([self._transform(document) for document in documents])
        self.svd = self.svd.fit(encoding)
        return self
    
    def _transform(self, document):
        unknown_vec = np.zeros(self.n_components_embedding)
        if len(document) == 0:
            return unknown_vec 
        words = document.lower().split()
        vec = [self.embeddings_dict[word].reshape(1,-1)  if word in self.embeddings_dict else unknown_vec.reshape(1,-1) for word in words]       
        encoding = np.vstack(vec)
        # multiply by tfidf
        tfidf_vec = text_to_tfidf(document, self.vectorizer)
        encoding =  (tfidf_vec.T * encoding.T).T
        encoding = encoding.sum(axis=0).reshape(-1)
        if self.normalize:
            encoding = normalize_func(encoding.reshape(1, -1)).reshape(-1)
        return encoding

    def transform(self, documents):
        encoding = np.vstack([self._transform(document) for document in documents])
        X = self.svd.transform(encoding)
        return X
    
    def fit_transform(self, documents, targets=None):
        return self.fit(documents, targets).transform(documents)


    
class CompositeTextVectorizer(object):
    def __init__(self, vectorizers_list=None, n_components=None, normalize=True):
        self.vectorizers_list = vectorizers_list
        self.vectorizers_are_fit = False
        self.n_components = n_components
        self.normalize = normalize
        if self.n_components is not None:
            self.svd = TruncatedSVD(n_components=self.n_components)
        else:
            self.svd = None
    
    def set_pretrained_vectorizers(self, vectorizers_list):
        self.vectorizers_list = vectorizers_list
        self.vectorizers_are_fit = True
        return self
        
    def fit(self, documents, targets=None):
        if self.vectorizers_are_fit is False:
            self.vectorizers_list = [vectorizer.fit(documents, targets) for vectorizer in self.vectorizers_list]
            self.vectorizers_are_fit = True
        if self.svd is not None:
            X = self._transform(documents)
            self.svd = self.svd.fit(X)
        return self
    
    def _transform(self, documents):
        X = np.hstack([vectorizer.transform(documents) for vectorizer in self.vectorizers_list])
        return X

    def transform(self, documents):
        X = self._transform(documents)
        if self.svd is not None:
            X = self.svd.transform(X)
        if self.normalize:
            X = normalize_func(X)
        return X
    
    def fit_transform(self, documents, targets=None):
        return self.fit(documents, targets).transform(documents)
    