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
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

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
data = data.copy().reset_index(drop=True)


''' EXTRACT GENRES '''

movie_genres = data[['original_title', 'overview', 'genres', 'vote_average']]
movie_genres = movie_genres.copy()
genres = pd.read_csv("genres_collection.csv", sep=',')
sorted_unique_genres = genres['genre'].tolist()

movie_genres['genres'] = movie_genres['genres'].apply(lambda x: [i[1] for i in x])
movie_genres['genres'] = movie_genres['genres'].apply(", ".join)

# increase the weight for the genres by 3 times compared to the movie overview
movie_genres['overview_genres'] = movie_genres['overview'] + movie_genres['genres'] + movie_genres['genres'] + movie_genres['genres']


''' PROCESS TEXT '''

movie_genres['overview'] = movie_genres['overview'].astype(str)
# movie_genres['overview'] = movie_genres['overview'].apply(clean_text)
tfidfvectorizer = TfidfVectorizer(analyzer=clean_text, stop_words='english')
overview_bow = tfidfvectorizer.fit_transform(movie_genres['overview'])
# print(overview_bow)
# print(type(overview_bow))
# overview_bow = overview_bow.todense()
# print(type(overview_bow))


"""

**************** ANALYZING ****************

"""


''' Build a ML model to predict movie rating based on overview and genres '''
X_train, X_valid, y_train, y_valid = train_test_split(overview_bow, movie_genres['vote_average'])

print("KNeighbors Regressor:")
model_kneighbors = KNeighborsRegressor(15)
model_kneighbors.fit(X_train, y_train)
print(model_kneighbors.score(X_train, y_train))
print(model_kneighbors.score(X_valid, y_valid))
y_pred = model_kneighbors.predict(X_valid)
print(r2_score(y_valid, y_pred))
print("")

print("Random Forest Regressor:")
model_forest = RandomForestRegressor(1000, max_depth=5, min_samples_leaf=100, random_state=50)
model_forest.fit(X_train, y_train)
print(model_forest.score(X_train, y_train))
print(model_forest.score(X_valid, y_valid))
y_pred = model_kneighbors.predict(X_valid)
print(r2_score(y_valid, y_pred))
print("")

# https://machinelearningmastery.com/configure-gradient-boosting-algorithm/
print("Gradient Boost Regressor:")
model_gradient_boost = GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_leaf=6)
model_gradient_boost.fit(X_train, y_train)
print(model_gradient_boost.score(X_train, y_train))
print(model_gradient_boost.score(X_valid, y_valid))
y_pred = model_kneighbors.predict(X_valid)
print(r2_score(y_valid, y_pred))
print("")

print("Support Vector Machine Regressor:")
model_svr = SVR(kernel='rbf')
model_svr.fit(X_train, y_train)
print(model_svr.score(X_train, y_train))
print(model_svr.score(X_valid, y_valid))
y_pred = model_kneighbors.predict(X_valid)
print(r2_score(y_valid, y_pred))
print("")

print("Neural Network Regressor:")
model_neural = MLPRegressor(hidden_layer_sizes=(5, 4), activation='logistic', solver='lbfgs', max_iter=36000)
model_neural.fit(X_train, y_train)
print(model_neural.score(X_train, y_train))
print(model_neural.score(X_valid, y_valid))
y_pred = model_kneighbors.predict(X_valid)
print(r2_score(y_valid, y_pred))
print("")

print("Voting Regressor:")
model_voting = VotingRegressor([
                            ('neighbors', KNeighborsRegressor(50)),
                            ('forest', RandomForestRegressor(n_estimators=100, min_samples_leaf=20)),
                            ('svr', SVR(kernel='rbf')),
                            ('neural', MLPRegressor(hidden_layer_sizes=(4, 5), activation='logistic', solver='lbfgs', max_iter=50000))
                            ]
)

model_voting.fit(X_train, y_train)
print(model_voting.score(X_train, y_train))
print(model_voting.score(X_valid, y_valid))
y_pred = model_kneighbors.predict(X_valid)
print(r2_score(y_valid, y_pred))
print("")

# data.to_csv(path_or_buf="output.csv", index=False)
