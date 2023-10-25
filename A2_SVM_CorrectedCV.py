from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, f1_score


st = stopwords.words('english')
stemmer = PorterStemmer()

def preprocess_text(raw_text):
    words = [stemmer.stem(w) for w in raw_text.lower().split() if w not in st]
    return (" ".join(words))


train_data = pd.read_csv("/Users/peibo1/Desktop/BMI 550/Assignment 2/fallreports_2023-9-21_train.csv")
test_data = pd.read_csv("/Users/peibo1/Desktop/BMI 550/Assignment 2/fallreports_2023-9-21_test.csv")


train_data = train_data.dropna(subset=['fall_description'])
test_data = test_data.dropna(subset=['fall_description'])

train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)


X_train = train_data['fall_description']
Y_train = train_data['fog_q_class']

X_test = test_data['fall_description']
Y_test = test_data['fog_q_class']

training_texts_preprocessed = [preprocess_text(tr) for tr in X_train]
test_texts_preprocessed = [preprocess_text(te) for te in X_test]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, preprocessor=None, max_features=2000)

grid_params = {
    'C': [1, 2, 4, 16, 32, 64],
    'kernel': ['rbf', 'linear']
}

svm_classifier = SVC()

grid_search = GridSearchCV(svm_classifier, grid_params, scoring='accuracy', cv=skf)

results = []

for train_index, test_index in skf.split(training_texts_preprocessed, Y_train):
    training_texts_preprocessed_train = [training_texts_preprocessed[i] for i in train_index]
    training_texts_preprocessed_dev = [training_texts_preprocessed[i] for i in test_index]

    ttp_train, ttp_test = Y_train.iloc[train_index], Y_train.iloc[test_index]

    training_data_vectors = vectorizer.fit_transform(training_texts_preprocessed_train).toarray()
    test_data_vectors = vectorizer.transform(training_texts_preprocessed_dev).toarray()

    grid_search.fit(training_data_vectors, ttp_train)
    
    results.append({
        "best_params": grid_search.best_params_,
        "cv_results": grid_search.cv_results_
    })


best_mean_score = -1
best_params = None

for result in results:
    if result['cv_results']['mean_test_score'].max() > best_mean_score:
        best_mean_score = result['cv_results']['mean_test_score'].max()
        best_params = result['best_params']

vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, preprocessor=None, max_features=2000)
X_train_vectorized = vectorizer.fit_transform(training_texts_preprocessed).toarray()
best_model = SVC(C=best_params['C'], kernel=best_params['kernel'])
best_model.fit(X_train_vectorized, Y_train)

X_test_vectorized = vectorizer.transform(test_texts_preprocessed).toarray()
predictions = best_model.predict(X_test_vectorized)

accuracy = accuracy_score(Y_test, predictions)
micro_f1 = f1_score(Y_test, predictions, average='micro')
macro_f1 = f1_score(Y_test, predictions, average='macro')

best_params, accuracy, micro_f1, macro_f1



