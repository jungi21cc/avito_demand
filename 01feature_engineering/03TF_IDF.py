import numpy as np
import pandas as pd
from scipy import sparse

train = pd.read_csv('../train1.csv')
test = pd.read_csv('../test1.csv')

for col in ['description', 'title']:
    df['num_words_' + col] = df[col].apply(lambda comment: len(comment.split()))
    df['num_unique_words_' + col] = df[col].apply(lambda comment: len(set(w for w in comment.split())))

df['words_vs_unique_title'] = df['num_unique_words_title'] / df['num_words_title'] * 100
df['words_vs_unique_description'] = df['num_unique_words_description'] / df['num_words_description'] * 100
df['num_desc_punct'] = df['description'].apply(lambda x: count(x, set(string.punctuation)))

count_vectorizer_title = CountVectorizer(stop_words=stopwords.words('russian'), lowercase=True, min_df=25)

title_counts = count_vectorizer_title.fit_transform(train['title'].append(test['title']))

train_title_counts = title_counts[:len(train)]
test_title_counts = title_counts[len(train):]


count_vectorizer_desc = TfidfVectorizer(stop_words=stopwords.words('russian'), 
                                        lowercase=True, ngram_range=(1, 2),
                                        max_features=15000)

desc_counts = count_vectorizer_desc.fit_transform(train['description'].append(test['description']))

train_desc_counts = desc_counts[:len(train)]
test_desc_counts = desc_counts[len(train):]

print(train_title_counts.shape, train_desc_counts.shape)


sparse.save_npz("X.npz", X)
X = sparse.load_npz("X.npz")


sparse.save_npz("testing.npz", testing)
testing = sparse.load_npz("testing.npz")
