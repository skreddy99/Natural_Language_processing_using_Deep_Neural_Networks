import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemm = WordNetLemmatizer()

n_hl1_nodes = 500
n_hl2_nodes = 500
n_classes = 2
hm_data = 1600000
hm_test_data = 1600000
n_epochs = 10
n_vocab=11682
learning_rt=0.01

x = tf.placeholder(tf.float32, shape=[None, n_vocab])
y = tf.placeholder(tf.float32, shape=[None, n_classes])


hl1 = {'weight':tf.Variable(tf.random_normal([n_vocab, n_hl1_nodes])),
       'bias':tf.Variable(tf.random_normal([n_hl1_nodes]))}

hl2 = {'weight':tf.Variable(tf.random_normal([n_hl1_nodes, n_hl2_nodes])),
       'bias':tf.Variable(tf.random_normal([n_hl2_nodes]))}

ol = {'weight':tf.Variable(tf.random_normal([n_hl2_nodes, n_classes])),
      'bias':tf.Variable(tf.random_normal([n_classes]))}

def NN_model(data):
    l1 = tf.add(tf.matmul(data,hl1['weight']), hl1['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hl2['weight']), hl2['bias'])
    l2 = tf.nn.relu(l2)

    output = tf.add(tf.matmul(l2, ol['weight']), ol['bias'])
    return output

saver = tf.train.Saver()
tf_log = 'tf.log'

def train_NN(x):
    pred = NN_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rt, rho=0.95).minimize(cost)
    print("optimizer done")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            epoch = int(open(tf_log, 'r').read().split('\n')[-2])
            print('STARTING AT EPOCH:', epoch)
        except:
            epoch = 1

        while epoch <= n_epochs:
            print('\n\n'+'Epoch:', epoch)
            if epoch !=1:
                saver.restore(sess,'./model.ckpt')
                print("Picking the model from the ckpt")
            epoch_loss =0
            with open ('lexicon_big_dataset.pickle','rb') as f:
                lexicon = pickle.load(f)

            with open('train_set_shuffled.csv', buffering=200000, encoding='latin-1') as f:
                counter = 0
                for line in f:
                    counter +=1
                    label = line.split(':::')[0]
                    tweet = line.split(':::')[1]
                    current_words = word_tokenize(tweet.lower())
                    current_words = [lemm.lemmatize(i) for i in current_words]
                    features = np.zeros(len(lexicon))
                    for word in current_words:
                        if word.lower() in lexicon:
                            index_value = lexicon.index(word.lower())
                            features[index_value]+=1
                    batch_x = np.array([list(features)])
                    batch_y = np.array([eval(label)])
                    _, c = sess.run([optimizer, cost], feed_dict={x:np.array(batch_x),
                                                                      y:np.array(batch_y)})
                    epoch_loss+= c
                    if counter >= hm_data:
                        print("Reached ", hm_data, "data breaking")
                        break

            save_path = saver.save(sess, "./model.ckpt")
            print("Model saved in file: %s" % save_path)
            print('Epoch', epoch, 'completed out of ', n_epochs, 'loss:', epoch_loss)
            with open(tf_log,'a') as f:
                f.write(str(epoch)+ '\n')
            epoch += 1

train_NN(x)

def test_NN():
    pred = NN_model(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            try:
                saver.restore(sess,"./model.ckpt")
            except Exception as e:
                print(str(e))

        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        feature_sets = []
        labels = []
        counter = 0
        with open('processed-test-set.csv', buffering=200000) as f:
            for line in f:
                try:
                    features = list(eval(line.split('::')[0]))
                    label = list(eval(line.split('::')[1]))
                    feature_sets.append(features)
                    labels.append(label)
                    counter+=1
                except:
                    pass
                if counter >= hm_test_data:
                    print("Reached ", hm_test_data, " of test data; breaking")
                    break

        test_x = np.array(feature_sets)
        test_y = np.array(labels)
        print('Accuracy:', accuracy.eval(feed_dict={x: test_x, y: test_y}))
        print('\n\n','Tested',counter,'samples.')
        print('Layer1 nodes', n_hl1_nodes)
        print('Layer2 nodes', n_hl2_nodes)
        print('Tweets covered per batch', hm_data)
        print('Num of epochs', n_epochs)
        print('Size of vocab', n_vocab)
        print('Learning rate: 0.01')

test_NN()










