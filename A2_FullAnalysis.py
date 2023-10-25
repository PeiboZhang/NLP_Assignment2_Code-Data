import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from nltk.stem.porter import *
from nltk.corpus import stopwords
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# ------------------------------------------------------------------------
# Section 1: Data Preprocess, Features Engineering, Training&Test Split
# ------------------------------------------------------------------------


st = stopwords.words('english')
stemmer = PorterStemmer()

word_clusters = {}

def loadwordclusters():
    infile = open('/Users/peibo1/Desktop/BMI 550/Assignment 2/50mpaths2')
    for line in infile:
        items = str.strip(line).split()
        class_ = items[0]
        term = items[1]
        word_clusters[term] = class_
    return word_clusters

def getclusterfeatures(sent):
    sent = sent.lower()
    terms = nltk.word_tokenize(sent)
    cluster_string = ''
    for t in terms:
        if t in word_clusters.keys():
            cluster_string += 'clust_' + word_clusters[t] + '_clust '
    return str.strip(cluster_string)

def preprocess_text(raw_text):
    words = [stemmer.stem(w) for w in raw_text.lower().split() if w not in st]
    return (" ".join(words))


train_data = pd.read_csv('/Users/peibo1/Desktop/BMI 550/Assignment 2/fallreports_2023-9-21_train.csv')
train_has_na = train_data['fall_description'].isna().sum()
print(train_has_na) # 3 NA record in the fall description
train_data = train_data.dropna(subset=['fall_description'])
print(train_data['record_id'].nunique()) # 35 patients in the training dataset (26 data in the original dataset, but that patient does not have description value)


test_data = pd.read_csv('/Users/peibo1/Desktop/BMI 550/Assignment 2/fallreports_2023-9-21_test.csv')
has_na = test_data['fall_description'].isna().sum()
print(has_na) # 2 NA record in the fall description
test_data = test_data.dropna(subset=['fall_description'])
print(test_data['record_id'].nunique()) #25 patients in the test dataset (26 data in the original dataset, but that patient does not have description value)

word_clusters = loadwordclusters()

train_data['fall_description'] = train_data['fall_description'].apply(preprocess_text)
test_data['fall_description'] = test_data['fall_description'].apply(preprocess_text)

vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=1000)
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3),max_features=1000)
clustervectorizer = CountVectorizer(ngram_range=(1,3), max_features=1000)
pos_vectorizer = CountVectorizer(ngram_range=(1,3), max_features=1000)
length_encoder = OneHotEncoder(categories='auto', handle_unknown='ignore')

def get_postags(sent):
    tokens = nltk.word_tokenize(sent)
    pos_tags = nltk.pos_tag(tokens)
    return ' '.join([tag for word, tag in pos_tags])

train_data['pos_tag'] = train_data['fall_description'].apply(get_postags)
test_data['pos_tag'] = test_data['fall_description'].apply(get_postags)

def get_sent_length(sent):
    return len(nltk.word_tokenize(sent))

train_data['sent_length'] = train_data['fall_description'].apply(get_sent_length)
test_data['sent_length'] = test_data['fall_description'].apply(get_sent_length)

train_ngrams = vectorizer.fit_transform(train_data['fall_description']).toarray()
test_ngrams = vectorizer.transform(test_data['fall_description']).toarray()

train_tfidf = tfidf_vectorizer.fit_transform(train_data['fall_description']).toarray()
test_tfidf = tfidf_vectorizer.transform(test_data['fall_description']).toarray()

train_clusters = clustervectorizer.fit_transform(train_data['fall_description'].apply(getclusterfeatures)).toarray()
test_clusters = clustervectorizer.transform(test_data['fall_description'].apply(getclusterfeatures)).toarray()

train_pos = pos_vectorizer.fit_transform(train_data['pos_tag']).toarray()
test_pos = pos_vectorizer.transform(test_data['pos_tag']).toarray()

train_lengths = length_encoder.fit_transform(train_data[['sent_length']]).toarray()
test_lengths = length_encoder.transform(test_data[['sent_length']]).toarray()

X_train = np.concatenate((train_ngrams,train_tfidf,train_clusters,train_pos,train_lengths), axis=1)
X_test = np.concatenate((test_ngrams,test_tfidf,test_clusters,test_pos,test_lengths), axis=1)

y_train = train_data['fog_q_class'].values
y_test = test_data['fog_q_class'].values

# ------------------------------------------------------------------------
# Section 2: Different Text Classifiers
# ------------------------------------------------------------------------


#--------------------------Naive Bayes (Baseline) --------------------------#

nb_clf = MultinomialNB()

nb_clf.fit(X_train, y_train)

y_pred = nb_clf.predict(X_test)

nb_accuracy = accuracy_score(y_test, y_pred)
nb_f1_micro = f1_score(y_test, y_pred, average='micro')
nb_f1_macro = f1_score(y_test, y_pred, average='macro')

print("Naive Bayes - Accuracy:", nb_accuracy)
print("Naive Bayes - F1 (Micro):", nb_f1_micro)
print("Naive Bayes - F1 (Macro):", nb_f1_macro)


#--------------------------Logistic Regression--------------------------#


lr_clf = LogisticRegression(max_iter=1000)
lr_cv_scores = cross_val_score(lr_clf, X_train, y_train, cv=10, scoring='f1_micro')
print("Logistic Regression - Cross-validation F1 (Micro):", lr_cv_scores.mean())

from sklearn.model_selection import GridSearchCV

lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100]
}

lr_grid_search = GridSearchCV(LogisticRegression(max_iter=1000), lr_param_grid, cv=10, scoring='f1_micro')
lr_grid_search.fit(X_train, y_train)
print("Best Hyperparameters for Logistic Regression:", lr_grid_search.best_params_)

best_lr = lr_grid_search.best_estimator_
best_lr.fit(X_train, y_train)
lr_pred = best_lr.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_f1_micro = f1_score(y_test, lr_pred, average='micro')
lr_f1_macro = f1_score(y_test, lr_pred, average='macro')

print("Logistic Regression - Accuracy:", lr_accuracy)
print("Logistic Regression - F1 (Micro):", lr_f1_micro)
print("Logistic Regression - F1 (Macro):", lr_f1_macro)

#--------------------------Random Forest--------------------------#


rf_clf = RandomForestClassifier()
rf_cv_scores = cross_val_score(rf_clf, X_train, y_train, cv=10, scoring='f1_micro')
print("Random Forest - Cross-validation F1 (Micro):", rf_cv_scores.mean())

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf_grid_search = GridSearchCV(RandomForestClassifier(), rf_param_grid, cv=10, scoring='f1_micro')
rf_grid_search.fit(X_train, y_train)
print("Best Hyperparameters for Random Forest:", rf_grid_search.best_params_)

best_rf = rf_grid_search.best_estimator_
best_rf.fit(X_train, y_train)
rf_pred = best_rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1_micro = f1_score(y_test, rf_pred, average='micro')
rf_f1_macro = f1_score(y_test, rf_pred, average='macro')

print("Random Forest - Accuracy:", rf_accuracy)
print("Random Forest - F1 (Micro):", rf_f1_micro)
print("Random Forest - F1 (Macro):", rf_f1_macro)

#--------------------------SVM--------------------------#

svm_clf = SVC()
svm_cv_scores = cross_val_score(svm_clf, X_train, y_train, cv=10, scoring='f1_micro')
print("SVM - Cross-validation F1 (Micro):", svm_cv_scores.mean())

svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

svm_grid_search = GridSearchCV(SVC(), svm_param_grid, cv=10, scoring='f1_micro')
svm_grid_search.fit(X_train, y_train)
print("Best Hyperparameters for SVM:", svm_grid_search.best_params_)

best_svm = svm_grid_search.best_estimator_
best_svm.fit(X_train, y_train)
svm_pred = best_svm.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)
svm_f1_micro = f1_score(y_test, svm_pred, average='micro')
svm_f1_macro = f1_score(y_test, svm_pred, average='macro')

print("SVM - Accuracy:", svm_accuracy)
print("SVM - F1 (Micro):", svm_f1_micro)
print("SVM - F1 (Macro):", svm_f1_macro)


gb_clf = GradientBoostingClassifier()
gb_cv_scores = cross_val_score(gb_clf, X_train, y_train, cv=5, scoring='f1_micro')
print("Gradient Boosting - Cross-validation F1 (Micro):", gb_cv_scores.mean())

gb_param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [3, 5, 8]
}

gb_grid_search = GridSearchCV(GradientBoostingClassifier(), gb_param_grid, cv=5, scoring='f1_micro')
gb_grid_search.fit(X_train, y_train)
print("Best Hyperparameters for Gradient Boosting:", gb_grid_search.best_params_)

best_gb = gb_grid_search.best_estimator_
best_gb.fit(X_train, y_train)
gb_pred = best_gb.predict(X_test)

gb_accuracy = accuracy_score(y_test, gb_pred)
gb_f1_micro = f1_score(y_test, gb_pred, average='micro')
gb_f1_macro = f1_score(y_test, gb_pred, average='macro')

print("Gradient Boosting - Accuracy:", gb_accuracy)
print("Gradient Boosting - F1 (Micro):", gb_f1_micro)
print("Gradient Boosting - F1 (Macro):", gb_f1_macro)


#--------------------------Neural Network: MLP Classifier--------------------------#


nn_clf = MLPClassifier(max_iter=1000)
nn_cv_scores = cross_val_score(nn_clf, X_train, y_train, cv=5, scoring='f1_micro')
print("Neural Network - Cross-validation F1 (Micro):", nn_cv_scores.mean())

nn_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01]
}

nn_grid_search = GridSearchCV(MLPClassifier(max_iter=1000), nn_param_grid, cv=5, scoring='f1_micro')
nn_grid_search.fit(X_train, y_train)
print("Best Hyperparameters for Neural Network:", nn_grid_search.best_params_)

best_nn = nn_grid_search.best_estimator_
best_nn.fit(X_train, y_train)
nn_pred = best_nn.predict(X_test)

nn_accuracy = accuracy_score(y_test, nn_pred)
nn_f1_micro = f1_score(y_test, nn_pred, average='micro')
nn_f1_macro = f1_score(y_test, nn_pred, average='macro')

print("Neural Network - Accuracy:", nn_accuracy)
print("Neural Network - F1 (Micro):", nn_f1_micro)
print("Neural Network - F1 (Macro):", nn_f1_macro)

#--------------------------Ensemble Classifier: Voting--------------------------#


lr_clf = best_lr
rf_clf = best_rf
svm_clf = best_svm

voting_clf = VotingClassifier(estimators=[
    ('lr', lr_clf), 
    ('rf', rf_clf), 
    ('svm', svm_clf)
], voting='hard')

voting_clf.fit(X_train, y_train)

voting_pred = voting_clf.predict(X_test)
voting_accuracy = accuracy_score(y_test, voting_pred)
voting_f1_micro = f1_score(y_test, voting_pred, average='micro')
voting_f1_macro = f1_score(y_test, voting_pred, average='macro')

print("Voting Classifier - Accuracy:", voting_accuracy)
print("Voting Classifier - F1 (Micro):", voting_f1_micro)
print("Voting Classifier - F1 (Macro):", voting_f1_macro)


# ------------------------------------------------------------------------
# Section 3: Performance vs. Training Data Size
# ------------------------------------------------------------------------

sizes = np.linspace(0.1, 1, 10) 
f1_scores = []

for size in sizes:
    sample_size = int(size * X_train.shape[0])
    X_sample = X_train[:sample_size]
    y_sample = y_train[:sample_size]
    
    best_nn.fit(X_sample, y_sample)
    y_pred = best_nn.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='micro')
    f1_scores.append(f1)

plt.plot(sizes, f1_scores)
plt.xlabel('Fraction of Training Set')
plt.ylabel('F1 (Micro)')
plt.title('Training Set Size vs. Performance (MLP Classifier)')
plt.show()

# ------------------------------------------------------------------------
# Section 4: Ablation Study on the Best Performaned Classifier
# ------------------------------------------------------------------------

original_nn_f1_micro = nn_f1_micro
print("Original NN F1 (Micro):", original_nn_f1_micro)

#Without ngrams
X_train_without_ngrams = np.concatenate((train_tfidf, train_clusters, train_pos, train_lengths), axis=1)
X_test_without_ngrams = np.concatenate((test_tfidf, test_clusters, test_pos, test_lengths), axis=1)

best_nn.fit(X_train_without_ngrams, y_train)
nn_pred_without_ngrams = best_nn.predict(X_test_without_ngrams)
nn_f1_micro_without_ngrams = f1_score(y_test, nn_pred_without_ngrams, average='micro')
print("NN F1 (Micro) without N-grams:", nn_f1_micro_without_ngrams)

#Without tfidf
X_train_without_tfidf = np.concatenate((train_ngrams, train_clusters, train_pos, train_lengths), axis=1)
X_test_without_tfidf = np.concatenate((test_ngrams, test_clusters, test_pos, test_lengths), axis=1)

best_nn.fit(X_train_without_tfidf, y_train)
nn_pred_without_tfidf = best_nn.predict(X_test_without_tfidf)
nn_f1_micro_without_tfidf = f1_score(y_test, nn_pred_without_tfidf, average='micro')
print("NN F1 (Micro) without TF-IDF:", nn_f1_micro_without_tfidf)

#Without clusters
X_train_without_clusters = np.concatenate((train_ngrams, train_tfidf, train_pos, train_lengths), axis=1)
X_test_without_clusters = np.concatenate((test_ngrams, test_tfidf, test_pos, test_lengths), axis=1)

best_nn.fit(X_train_without_clusters, y_train)
nn_pred_without_clusters = best_nn.predict(X_test_without_clusters)
nn_f1_micro_without_clusters = f1_score(y_test, nn_pred_without_clusters, average='micro')
print("NN F1 (Micro) without Clusters:", nn_f1_micro_without_clusters)

#Without pos
X_train_without_pos = np.concatenate((train_ngrams, train_tfidf, train_clusters, train_lengths), axis=1)
X_test_without_pos = np.concatenate((test_ngrams, test_tfidf, test_clusters, test_lengths), axis=1)

best_nn.fit(X_train_without_pos, y_train)
nn_pred_without_pos = best_nn.predict(X_test_without_pos)
nn_f1_micro_without_pos = f1_score(y_test, nn_pred_without_pos, average='micro')
print("NN F1 (Micro) without Pos Tagging:", nn_f1_micro_without_pos)

#Without lengths
X_train_without_lengths = np.concatenate((train_ngrams, train_tfidf, train_clusters, train_pos), axis=1)
X_test_without_lengths = np.concatenate((test_ngrams, test_tfidf, test_clusters, test_pos), axis=1)

best_nn.fit(X_train_without_lengths, y_train)
nn_pred_without_lengths = best_nn.predict(X_test_without_lengths)
nn_f1_micro_without_lengths = f1_score(y_test, nn_pred_without_lengths, average='micro')
print("NN F1 (Micro) without Lengths:", nn_f1_micro_without_lengths)

