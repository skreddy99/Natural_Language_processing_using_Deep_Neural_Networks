'''

Sentiment analyses for large dataset is slighlt different.
The approcah to pre-process the data is very similar to what
I did in small data project i.e. in the file Sentiment_Analyses.py (for small dataset).
Here is the brief algorithm for pre-processing:

1. Get the statements sorted as positive and negative
2. Store list of positive and negative statements in separate files
and call them pos.txt and neg.txt
3. Next create a lexicon of the words in pos and neg statements.
To arrive at the lexicon, first tokenize the words in each file.
Later convert the words into lower case. Next eliminate those
words that occur too frequently and too rarely. The logic is
that in either case the impact of these words on the sentiment is less
4. Now convert each sentence of pos file into a vector of 1s and 0s
using the following logic: First create an empty sentence array of 0s, of
length same as the lexicon. Now read each sentence of the positive
file. For each word in the sentence, find if there is a
matching word in the lexicon. If yes, replace the 0 in the empty sentence array
with a 1 at the same index value as the matching word in lexicon.
At the end of processing of each sentence, the sentence array
will be a collection of 0s and 1s. Each sentence array is of the same
length, same as the lexicon and will be filled with 1s and 0s. This is
called sentence vector.
5. Append the classification identifier to
each sentence vector. The classification identifier is an array
[1,0] for a positive statement and [0,1] for a negative
statement. For example, a sentence vector for a positive sentence
will look like [....1,0,0,0,0,0,1....][1,0].
6. Repeat the same process for the neg file. The example of
a negative sentence vector will look like [....0,0,0,0,1,1,1....][0,1]
7. Merge these two list of sentence vectors into a single file and shuffle them.
8. Later divide the data set into training set and test set (10%).


'''


import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd

lemm = WordNetLemmatizer()

def extract_tweet_and_label(f_in, f_out):
    outfile = open(f_out,'a')
    with open(f_in, buffering=200000,encoding='latin-1') as f:
        try:
            for line in f:
                line = line.replace('"','')
                initial_polarity = line.split(',')[0]
                if initial_polarity == '0':
                    initial_polarity = [1,0]
                elif initial_polarity == '4':
                    initial_polarity = [0,1]

                tweet = line.split(',')[-1]
                outline = str(initial_polarity)+':::'+tweet
                outfile.write(outline)
        except Exception as e:
            print(str(e))
    outfile.close()

extract_tweet_and_label('training.1600000.processed.noemoticon.csv', 'training_large_dataset.csv')
extract_tweet_and_label('testdata.manual.2009.06.14.csv','test_large_dataset.csv')


def create_lexicon(f_in):
    lexicon = []
    with open(f_in,'r', buffering=100000, encoding='latin-1') as f:
        try:
            counter = 1
            content = ''
            for line in f:
                counter +=1
                if (counter/2500.0).is_integer():
                    tweet = line.split(':::')[1]
                    content+= ''+tweet
                    words = word_tokenize(content)
                    words = [lemm.lemmatize(i) for i in words]
                    lexicon = list(set(lexicon+words))
                    print(counter, len(lexicon))
        except Exception as e:
            print(str(e))
    with open('lexicon.pickle', 'wb') as f:
        pickle.dump(lexicon,f)

create_lexicon('training_large_dataset.csv')

def convert_to_vec(fin, fout, lexicon_pickle):
    with open(lexicon_pickle,'rb') as f:
        lexicon = pickle.load(f)
    outfile = open(fout,'a')
    with open(fin,buffering=20000, encoding='latin-1') as f:
        counter = 0
        for line in f:
            counter+=1
            label=line.split(':::')[0]
            tweet =line.split(':::')[1]
            current_words=word_tokenize(tweet.lower())
            current_words = [lemm.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))

            for word in current_words:
                if word.lower() in lexicon:
                    index_value=lexicon.index(word.lower())
                    features[index_value] +=1
            features = list(features)
            outline = str(features)+'::'+str(label)+'\n'
            outfile.write(outline)
        print(counter)

convert_to_vec('test_large_dataset.csv', 'processed-test-set.csv', 'lexicon.pickle')

def shuffle_data(f_in):
    df = pd.read_csv(f_in, error_bad_lines=False)
    df = df.iloc[np.random.permutation(len(df))]
    print(df.head)
    df.to_csv('train_set_shuffled.csv', index=False)

shuffle_data('training_large_dataset.csv')



def create_test_data_pickle(fin):
    feature_sets =[]
    labels = []
    counter = 0
    with open(fin, buffering=20000) as f:
        for line in f:
            try:
                features = list(eval(line.split('::')[0]))
                label = list(eval(line.split('::')[1]))
                feature_sets.append(features)
                labels.append(label)
                counter +=1
            except:
                pass
    print(counter)
    feature_sets = np.array(feature_sets)
    labels = np.array(labels)

create_test_data_pickle('processed-test-set.csv')

