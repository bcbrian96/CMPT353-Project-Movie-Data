import ast
import string
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import stem
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)


def clean_text(text):
    text = text.lower()
    not_punctuation = [char for char in text if char not in string.punctuation]
    not_punctuation = ''.join(not_punctuation)
    cleaned = [word for word in not_punctuation.split() if word not in stopwords]
    cleaned = ([stemmer.stem(word) for word in cleaned])

    return cleaned

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

mlb = MultiLabelBinarizer()

''' EXTRACT GENRES '''

movie_genres = data[['original_title', 'overview', 'popularity', 'genres']]
movie_genres = movie_genres.copy()
genres = pd.read_csv("genres_collection.csv", sep=',')
sorted_unique_genres = genres['genre'].tolist()

movie_genres['genres'] = movie_genres['genres'].apply(lambda x: [i[1] for i in x])
genres_binarized = pd.DataFrame(mlb.fit_transform(movie_genres['genres']), columns=mlb.classes_, index=movie_genres.index)
movie_genres = movie_genres.join(genres_binarized)
movie_genres = movie_genres.drop(['genres'], axis=1)
print(movie_genres)

# grouped_data = movie_genres.groupby(sorted_unique_genres).agg(genres_avg=('rating', 'mean')).reset_index()
# # grouped_data['group'] = np.arange(len(grouped_data))
# prepared_data = movie_genres.merge(grouped_data, on=sorted_unique_genres)


''' PROCESS TEXT '''

movie_genres['overview'] = movie_genres['overview'].astype(str)
# Lancaster is strictest and PorterStemmer is least strict.
# Snowball is in the middle, so we use Snowball here for stemming text.
stemmer = stem.SnowballStemmer('english')
stopwords = set(stopwords.words('english'))
tfidfvectorizer = TfidfVectorizer(analyzer=clean_text, stop_words='english')
overview_bow = tfidfvectorizer.fit_transform(movie_genres['overview'])
print(overview_bow)
print(type(overview_bow))
overview_bow = overview_bow.todense()
print(type(overview_bow))

"""

**************** ANALYZING ****************

"""


plt.rcParams["figure.figsize"] = (18, 9)
# plt.hist(prepared_data['group'], bins=len(grouped_data))  # frequency of each group of genres
# plt.plot(prepared_data['group'], prepared_data['genres_avg'], 'bo', markersize=2)  # plot genre groups and their avg ratings
# plt.show()


''' Build a ML model to predict movie popularity based on overview and genres '''
col_names = ['overview']
col_names = col_names + sorted_unique_genres
features = movie_genres[col_names]
popularity = movie_genres['popularity']
X_train, X_valid, y_train, y_valid = train_test_split(overview_bow, popularity)
# X_train, X_valid, y_train, y_valid = train_test_split(overview_bow, genres_binarized)

# print("KNeighbors Regressor:")
# model_kneighbors = KNeighborsRegressor(15)
# model_kneighbors.fit(X_train, y_train)
# print(model_kneighbors.score(X_train, y_train))
# print(model_kneighbors.score(X_valid, y_valid))
# print("")

# This one gives low score of ~ 0.30 - 0.35
# print("Random Forest Regressor:")
# model_forest = RandomForestRegressor(1000, max_depth=5, min_samples_leaf=100, random_state=50)
# model_forest.fit(X_train, y_train)
# print(model_forest.score(X_train, y_train))
# print(model_forest.score(X_valid, y_valid))
# print("")

# https://machinelearningmastery.com/configure-gradient-boosting-algorithm/
# print("Gradient Boost Regressor:")
# model_gradient_boost = GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_leaf=6)
# model_gradient_boost.fit(X_train, y_train)
# print(model_gradient_boost.score(X_train, y_train))
# print(model_gradient_boost.score(X_valid, y_valid))
# print("")

# print("Support Vector Machine Regressor:")
# model_svr = SVR(kernel='rbf')
# model_svr.fit(X_train, y_train)
# print(model_svr.score(X_train, y_train))
# print(model_svr.score(X_valid, y_valid))
# print("")

# print("Neural Network Regressor:")
# model_neural = make_pipeline(StandardScaler(),
#                              MLPRegressor(hidden_layer_sizes=(5,4), activation='logistic', solver='lbfgs',
#                                           max_iter=36000))
# model_neural.fit(X_train, y_train)
# print(model_neural.score(X_train, y_train))
# print(model_neural.score(X_valid, y_valid))
# print("")

print("Voting Regressor:")
model_voting = make_pipeline(StandardScaler(),
                             VotingRegressor([
                            ('gaussian', GaussianProcessRegressor()),
                            ('neighbors', KNeighborsRegressor(50)),
                            ('forest', RandomForestRegressor(n_estimators=100, min_samples_leaf=20)),
                            ('svr', SVR()),
                            ('neural', MLPRegressor(hidden_layer_sizes=(4, 5), activation='logistic', solver='lbfgs', max_iter=36000))
                            ])
)

model_voting.fit(X_train, y_train)
print(model_voting.score(X_train, y_train))
print(model_voting.score(X_valid, y_valid))
print("")
#
# results = prepared_data.copy()
# results['predict'] = model_voting.predict(results[col_names])
# results['diff_from_truth'] = np.abs(results['predict'] - results['rating'])
# print(results)

# data.to_csv(path_or_buf="output.csv", index=False)
