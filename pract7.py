
from spacy.lang.en import English

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()

text = """He determined to drop his litigation with the monastry, and relinguish his claims to the wood-cuting and 
fishery rihgts at once. He was the more ready to do this becuase the rights had become much less valuable, and he had 
indeed the vaguest idea where the wood and river in question were."""

#  "nlp" Object is used to create documents with linguistic annotations.
my_doc = nlp(text)

# Create list of word tokens
token_list = []
for token in my_doc:
    token_list.append(token.text)

from spacy.lang.en.stop_words import STOP_WORDS

# Create list of word tokens after removing stopwords
filtered_sentence =[]

for word in token_list:
    lexeme = nlp.vocab[word]
    if lexeme.is_stop == False:
        filtered_sentence.append(word) 
print(token_list)
print("\n\n", filtered_sentence)


#New

import nltk
nltk.download("punkt")
nltk.download("stopword")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")



from nltk import word_tokenize, sent_tokenize
corpus = "Sachin was the GOAT of the previous generation, Virat is the GOAT of the this generation, Shubman will be the GOAT of the next generation"

print(word_tokenize(corpus))
print(sent_tokenize(corpus))


from nltk import pos_tag

tokens = word_tokenize(corpus)
print(pos_tag(tokens))


from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

tokens = word_tokenize(corpus)
cleaned_tokens = []
for token in tokens:
if (token not in stop_words):
cleaned_tokens.append(token)
print(cleaned_tokens)


from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

stemmed_tokens = []
for token in cleaned_tokens:
    stemmed = stemmer.stem(token)
    stemmed_tokens.append(stemmed)
print(stemmed_tokens)

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

lemmatized_tokens = []
for token in cleaned_tokens:
    lemmatized = lemmatizer.lemmatize(token)
    lemmatized_tokens.append(lemmatized)
print(lemmatized_tokens)


from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "Sachin was the GOAT of the previous generation",
    "Virat is the GOAT of the this generation",
    "Shubman will be the GOAT of the next generation"
]


vectorizer = TfidfVectorizer()

matrix = vectorizer.fit(corpus)
matrix.vocabulary_

tfidf_matrix = vectorizer.transform(corpus)
print(tfidf_matrix)

print(vectorizer.get_feature_names_out())

