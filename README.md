# NLP


# Assignment 1- Use of named entity recognition information extraction technique. <br>
- code :
```
import spacy
from spacy import displacy

# Load the small English NLP model
nlp = spacy.load("en_core_web_sm")

def spacy_ner(text):
    # Replace newline characters with spaces
    text = text.replace('\n', ' ')
    doc = nlp(text)

    entities = []
    labels = []
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE']:
            entities.append(ent)
            labels.append(ent.label_)

    return entities, labels

def fit_ner(texts):
    """Accepts a list of text strings instead of a DataFrame."""
    print('Fitting Spacy NER model...')

    ner = [spacy_ner(text) for text in texts]
    ner_org = {}
    ner_per = {}
    ner_gpe = {}

    for x in ner:
        for entity, label in zip(x[0], x[1]):
            if label == 'ORG':
                ner_org[entity.text] = ner_org.get(entity.text, 0) + 1
            elif label == 'PERSON':
                ner_per[entity.text] = ner_per.get(entity.text, 0) + 1
            elif label == 'GPE':
                ner_gpe[entity.text] = ner_gpe.get(entity.text, 0) + 1

    return {'ORG': ner_org, 'PER': ner_per, 'GPE': ner_gpe}

# Example static data (list of strings)
texts = [
    "Apple is looking to buy a startup in the United Kingdom.",
    "Elon Musk is the CEO of Tesla.",
    "Google is headquartered in Mountain View, California.",
    "The United States is a large country."
]

# Run the fit_ner function on the static data
named_entities = fit_ner(texts)

# Print the results
print("Organization Named Entities:", named_entities['ORG'])
print("Person Named Entities:", named_entities['PER'])
print("Geopolitical Entity Named Entities:", named_entities['GPE'])

```
if any error caused run before above code<br>
![image](https://github.com/user-attachments/assets/65a45959-b918-44a0-a82f-42c9c6a96673)


# Assignement 2-Implement sentiment analysis technique for classifying the data in to positive, negative or neutral class <br>
step1 - create a file named as ```sentimentanalysis.txt``` and paste the below data 

```
type=strongsubj len=1 word=abandon pos=verb stemmed1=y priorpolarity=negative
type=weaksubj len=1 word=able pos=adj stemmed1=y priorpolarity=positive
type=weaksubj len=1 word=abnormal pos=adj stemmed1=y priorpolarity=negative
type=strongsubj len=1 word=abuse pos=noun stemmed1=n priorpolarity=negative
type=weaksubj len=1 word=amazing pos=adj stemmed1=n priorpolarity=positive
```
step 2 
- code :
```
pos_words = []
neg_words = []

with open('sentimentanalysis.txt') as file:
    for line in file:
        line_attrib = line.split()
        word = line_attrib[2].split('=')[1]  # 2nd column in the file
        polarity = line_attrib[-1].split('=')[1]  # last column in the file
        if polarity == 'positive':
            pos_words.append(word)
        elif polarity == 'negative':
            neg_words.append(word)

# Print the counts after processing all lines
print('Total positive words found: ', len(pos_words))
print('Total negative words found: ', len(neg_words))
# Write results to file for future use
with open('pos_words.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(pos_words))

with open('neg_words.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(neg_words))

```

# Assignment 3 - Use of Natural Language Processing technique for text summarization
- code :
```
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample text
doc1 = """Python is a high-level, general-purpose programming language. Its design philosophy emphasizes
code readability with the use of significant indentation.
Python is dynamically-typed and garbage-collected. It supports multiple
programming paradigms, including
structured (particularly procedural), object-oriented, and functional
programming. It is often described
as a "batteries included" language due to its comprehensive standard
library.
Guido van Rossum began working on Python in the late 1980s as a
successor to the ABC programming
language and first released it in 1991 as Python 0.9.0. Python 2.0 was
released in 2000 and introduced new
features such as list comprehensions, cycle-detecting garbage collection,
reference counting, and Unicode
support.
Python 3.0, released in 2008, was a major revision that is not completely
backward-compatible with earlier
versions. Python 2 was discontinued with version 2.7.18 in 2020."""

# Process the text with spaCy
docx = nlp(doc1)

# Tokenization and word frequency calculation
stopwords = list(STOP_WORDS)
word_frequencies = {}
for word in docx:
    if word.text.lower() not in stopwords and word.text not in punctuation:
        if word.text.lower() not in word_frequencies:
            word_frequencies[word.text.lower()] = 1
        else:
            word_frequencies[word.text.lower()] += 1

# Normalize word frequencies by dividing by the max frequency
max_freq = max(word_frequencies.values())
for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word] / max_freq

# Sentence scoring based on word frequencies
sentence_scores = {}
for sent in docx.sents:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentence_scores:
                sentence_scores[sent] = word_frequencies[word.text.lower()]
            else:
                sentence_scores[sent] += word_frequencies[word.text.lower()]

# Selecting the top 30% of sentences based on their scores
from heapq import nlargest
select_length = int(len(sentence_scores) * 0.3)
summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)

# Join the selected sentences into a final summary
final_summary = [word.text for word in summary]
summary = ' '.join(final_summary)

print("Original Text Length:", len(doc1.split()))
print("Summary Length:", len(summary.split()))
print("\nSummary:\n", summary)
```

# Assignment 4 -Implement Simple Machine translation from one language to another.
- code :
```
from transformers import MarianMTModel, MarianTokenizer

# Load the model and tokenizer for English to Marathi translation
model_name = 'Helsinki-NLP/opus-mt-en-mr'  # Model for English to Marathi translation
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text, src_lang='en', tgt_lang='mr'):
    # Prepare the text input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Generate translation using the model
    translated_tokens = model.generate(**inputs)
    # Decode the tokens to get the translated text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# Sample text for translation
text_to_translate = "Hello, how are you?"

# Translate the text from English to Marathi
translated_text = translate_text(text_to_translate)

print(f"Original Text: {text_to_translate}")
print(f"Translated Text: {translated_text}")

```

# Assignment 5 -Implement a code for aspect mining and topic modeling
- code :
```
import spacy
from textblob import TextBlob

# Load the spacy model for English
sp = spacy.load("en_core_web_sm")

# Creating a list of positive and negative sentences.
mixed_sen = [
    'This chocolate truffle cake is really tasty',
    'This party is amazing!',
    'My mom is the best!',
    'App response is very slow!',
    'The trip to India was very enjoyable'
]

# An empty list for obtaining the extracted aspects from sentences.
ext_aspects = []

# Performing Aspect Extraction
for sen in mixed_sen:
    important = sp(sen)
    descriptive_item = ''
    target = ''

    for token in important:
        if token.dep_ == 'nsubj' and token.pos_ == 'NOUN':
            target = token.text
        if token.pos_ == 'ADJ':
            added_terms = ''
            for mini_token in token.children:
                if mini_token.pos_ != 'ADV':
                    continue
                added_terms += mini_token.text + ' '
            descriptive_item = added_terms + token.text

    ext_aspects.append({'aspect': target, 'description': descriptive_item})

print("ASPECT EXTRACTION\n")
print(ext_aspects)

# Associating Sentiment
for aspect in ext_aspects:
    aspect['sentiment'] = TextBlob(aspect['description']).sentiment

print("\nSENTIMENT ASSOCIATION\n")
print(ext_aspects)

print("")
print("")

import spacy
import gensim
import gensim.corpora as corpora
from nltk.corpus import stopwords
from pprint import pprint
import nltk

# Download stopwords
nltk.download('stopwords')

# Load spacy model for lemmatization
nlp = spacy.load("en_core_web_sm")

# Sample data for topic modeling
documents = [
    'This chocolate truffle cake is really tasty',
    'The party was amazing and everyone enjoyed it!',
    'My mom is the best and she loves me so much',
    'The app response is very slow, and it frustrates me',
    'The trip to India was very enjoyable and the experience was unforgettable',
]

# 1. Preprocessing (tokenization, stopwords removal, lemmatization)
stop_words = set(stopwords.words('english'))

def preprocess(doc):
    # Tokenize and lemmatize
    doc = nlp(doc)
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text.lower() not in stop_words]
    return tokens

processed_docs = [preprocess(doc) for doc in documents]

# 2. Create Dictionary and Corpus
# Create a dictionary representation of the documents
id2word = corpora.Dictionary(processed_docs)

# Create the Bag of Words corpus
corpus = [id2word.doc2bow(text) for text in processed_docs]

# 3. Applying LDA Model (Topic Modeling)
lda_model = gensim.models.LdaMulticore(corpus, id2word=id2word, num_topics=3, passes=10, workers=2, random_state=42)

# 4. Output the topics
pprint(lda_model.print_topics())

# Show the topic distribution for each document
for i, topic_distribution in enumerate(lda_model[corpus]):
    print(f"\nDocument {i + 1} Topic Distribution:")
    print(topic_distribution)
```


# Assignment 6 - Practical Python Implementation of Advanced Tokenization Techniques

- code : 
```
# Importing necessary libraries
from transformers import BertTokenizer, GPT2Tokenizer, RobertaTokenizer
import sentencepiece as spm
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Sample text
text = "Natural language processing (NLP) is a crucial technology for modern applications like chatbots, translation, and AI."

# 1. Basic Word Tokenization (NLTK)
print("Basic Word Tokenization (NLTK):")
word_tokens = word_tokenize(text)
print(word_tokens)

# 2. Subword Tokenization (BERT's WordPiece Tokenizer)
print("\nSubword Tokenization (BERT - WordPiece):")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_tokens = bert_tokenizer.tokenize(text)
print(bert_tokens)

# 3. Byte Pair Encoding (BPE) with GPT-2 Tokenizer
print("\nByte Pair Encoding (GPT-2):")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokens = gpt2_tokenizer.tokenize(text)
print(gpt2_tokens)

# 4. SentencePiece Tokenization (Pretrained on RoBERTa)
print("\nSentencePiece Tokenization (RoBERTa):")
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_tokens = roberta_tokenizer.tokenize(text)
print(roberta_tokens)

# 5. Train your own SentencePiece tokenizer (for custom data)
print("\nCustom SentencePiece Tokenizer (Training):")

# You would typically train on large text data, but here we simulate with small data
sample_data = "Natural language processing is essential for modern AI applications."
with open("sample_text.txt", "w") as f:
    f.write(sample_data)

# Train SentencePiece model with smaller vocabulary size
spm.SentencePieceTrainer.Train('--input=sample_text.txt --model_prefix=m --vocab_size=28')
sp = spm.SentencePieceProcessor(model_file='m.model')

# Tokenize using custom SentencePiece model
sentencepiece_tokens = sp.encode_as_pieces(text)
print(sentencepiece_tokens)
```

