import ast
import string
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import stem
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)


def clean_text(text):
    text = text.lower()

    # Lancaster is strictest and PorterStemmer is least strict.
    # Snowball is in the middle, so we use Snowball here for stemming text.
    stemmer = stem.SnowballStemmer('english')
    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(text)
    not_punctuation = [word for word in word_tokens if word not in string.punctuation]
    cleaned = [word for word in not_punctuation if word not in stop_words]
    cleaned_stem = [stemmer.stem(word) for word in cleaned]
    cleaned_join = " ".join(cleaned_stem)

    return cleaned_join


# read file
filename = "movies.csv"
data = pd.read_csv(filename, sep=',', converters={'genres': ast.literal_eval})


"""

**************** CLEANING AND ETL ****************

"""

data = data[data['genres'].apply(len).gt(0)]  # exclude observations with no genres recorded
data = data[data['overview'].notna()]  # exclude observations with no overview recorded
# data = data[data['keywords'].apply(len).gt(0)]  # exclude observations with no keywords recorded
# data = data[data['production_companies'].apply(len).gt(0)]  # exclude observations with no production_companies recorded
data = data.copy().reset_index(drop=True)


''' EXTRACT GENRES '''

movie_genres = data[['original_title', 'overview', 'popularity', 'genres']]
movie_genres = movie_genres.copy()
genres = pd.read_csv("genres_collection.csv", sep=',')
sorted_unique_genres = genres['genre'].tolist()

movie_genres['genres'] = movie_genres['genres'].apply(lambda x: [i[1] for i in x])
mlb = MultiLabelBinarizer()
genres_binarized = pd.DataFrame(mlb.fit_transform(movie_genres['genres']), columns=mlb.classes_, index=movie_genres.index)
movie_genres = movie_genres.join(genres_binarized)
movie_genres = movie_genres.drop(['genres'], axis=1)
print(movie_genres)

# grouped_data = movie_genres.groupby(sorted_unique_genres).agg(genres_avg=('rating', 'mean')).reset_index()
# # grouped_data['group'] = np.arange(len(grouped_data))
# prepared_data = movie_genres.merge(grouped_data, on=sorted_unique_genres)


''' PROCESS TEXT '''

movie_genres['overview'] = movie_genres['overview'].astype(str)
movie_genres['overview'] = movie_genres['overview'].apply(clean_text)
print(movie_genres)
# tfidfvectorizer = TfidfVectorizer(analyzer=clean_text, stop_words='english')
# overview_bow = tfidfvectorizer.fit_transform(movie_genres['overview'])


"""

**************** ANALYZING ****************

"""


plt.rcParams["figure.figsize"] = (18, 9)
# plt.hist(prepared_data['group'], bins=len(grouped_data))  # frequency of each group of genres
# plt.plot(prepared_data['group'], prepared_data['genres_avg'], 'bo', markersize=2)  # plot genre groups and their avg ratings
# plt.show()


''' Build a ML model to predict movie popularity based on overview and genres '''

X_train, X_valid, y_train, y_valid = train_test_split(movie_genres['overview'], movie_genres[sorted_unique_genres])
X_train_arr = np.array(X_train)
X_valid_arr = np.array(X_valid)

print("OVR - SGD Classifier:")
model_ovr_sgd = make_pipeline(
    CountVectorizer(),
    TfidfTransformer(),
    OneVsRestClassifier(SGDClassifier(loss="log", class_weight="balanced"))
)
model_ovr_sgd.fit(X_train, y_train)
y_pred = model_ovr_sgd.predict(X_valid_arr)
print(model_ovr_sgd.score(X_train, y_train))
print(model_ovr_sgd.score(X_valid, y_valid))
print(classification_report(y_valid, y_pred, target_names=sorted_unique_genres, zero_division=0))
print(precision_score(y_valid, y_pred, average="macro", zero_division=0))
print("")

print("OVR - Kneighbors Classifier:")
model_ovr_knc = make_pipeline(
    CountVectorizer(),
    TfidfTransformer(),
    OneVsRestClassifier(KNeighborsClassifier(15))
)
model_ovr_knc.fit(X_train, y_train)
y_pred = model_ovr_knc.predict(X_valid_arr)
print(model_ovr_knc.score(X_train, y_train))
print(model_ovr_knc.score(X_valid, y_valid))
print(classification_report(y_valid, y_pred, target_names=sorted_unique_genres, zero_division=0))
print(precision_score(y_valid, y_pred, average="macro", zero_division=0))
print("")

print("OVR - Random Forest Classifier:")
model_ovr_forest = make_pipeline(
    CountVectorizer(),
    TfidfTransformer(),
    OneVsRestClassifier(SGDClassifier(loss="log", class_weight="balanced"))
)
model_ovr_forest.fit(X_train, y_train)
y_pred = model_ovr_forest.predict(X_valid_arr)
print(model_ovr_forest.score(X_train, y_train))
print(model_ovr_forest.score(X_valid, y_valid))
print(classification_report(y_valid, y_pred, target_names=sorted_unique_genres, zero_division=0))
print(precision_score(y_valid, y_pred, average="macro", zero_division=0))
print("")

# https://machinelearningmastery.com/configure-gradient-boosting-algorithm/
print("OVR - Gradient Boost Classifier:")
model_ovr_boost = make_pipeline(
    CountVectorizer(),
    TfidfTransformer(),
    OneVsRestClassifier(GradientBoostingClassifier(n_estimators=500, max_depth=4, min_samples_leaf=6))
)
model_ovr_boost.fit(X_train, y_train)
y_pred = model_ovr_boost.predict(X_valid_arr)
print(model_ovr_boost.score(X_train, y_train))
print(model_ovr_boost.score(X_valid, y_valid))
print(classification_report(y_valid, y_pred, target_names=sorted_unique_genres, zero_division=0))
print(precision_score(y_valid, y_pred, average="macro", zero_division=0))
print("")

print("OVR - SVC Classifier:")
model_ovr_svc = make_pipeline(
    CountVectorizer(),
    TfidfTransformer(),
    OneVsRestClassifier(SVC())
)
model_ovr_svc.fit(X_train, y_train)
y_pred = model_ovr_svc.predict(X_valid_arr)
print(model_ovr_svc.score(X_train, y_train))
print(model_ovr_svc.score(X_valid, y_valid))
print(classification_report(y_valid, y_pred, target_names=sorted_unique_genres, zero_division=0))
print(precision_score(y_valid, y_pred, average="macro", zero_division=0))
print("")

print("OVR - MLP Classifier:")
model_ovr_mlp = make_pipeline(
    CountVectorizer(),
    TfidfTransformer(),
    OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(5, 4), activation='logistic', solver='lbfgs', max_iter=36000))
)
model_ovr_mlp.fit(X_train, y_train)
y_pred = model_ovr_mlp.predict(X_valid_arr)
print(model_ovr_mlp.score(X_train, y_train))
print(model_ovr_mlp.score(X_valid, y_valid))
print(classification_report(y_valid, y_pred, target_names=sorted_unique_genres, zero_division=0))
print(precision_score(y_valid, y_pred, average="macro", zero_division=0))
print("")

print("OVR - Voting Classifier:")
model_ovr_voting = make_pipeline(
    CountVectorizer(),
    TfidfTransformer(),
    OneVsRestClassifier(VotingClassifier([
        ('neighbors', KNeighborsClassifier(30)),
        ('forest', RandomForestClassifier(n_estimators=100, min_samples_leaf=20)),
        ('svc', SVC()),
        ('neural', MLPClassifier(hidden_layer_sizes=(4, 5), activation='logistic', solver='lbfgs', max_iter=36000))
    ]))
)
model_ovr_voting.fit(X_train, y_train)
y_pred = model_ovr_voting.predict(X_valid)
print(model_ovr_voting.score(X_train, y_train))
print(model_ovr_voting.score(X_valid, y_valid))
print(classification_report(y_valid, y_pred, target_names=sorted_unique_genres, zero_division=0))
print(precision_score(y_valid, y_pred, average="macro", zero_division=0))
print("")

# data.to_csv(path_or_buf="output.csv", index=False)
