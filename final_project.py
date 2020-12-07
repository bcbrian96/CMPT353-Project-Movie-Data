import sys
import json
import pandas as pd
import re
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


def convert_json_genres(json_list):
    movie_genres = []
    for json_object in json_list:
        genres_id = [json_object['id'], json_object['name']]
        movie_genres.append(genres_id)

    return movie_genres


def convert_json_keywords(json_list):
    movie_keywords = []
    for json_object in json_list:
        keyword_id = [json_object['id'], json_object['name']]
        movie_keywords.append(keyword_id)

    return movie_keywords


def convert_json_productions(json_list):
    movie_productions = []
    for json_object in json_list:
        production_id = [json_object['id'], json_object['name']]
        movie_productions.append(production_id)

    return movie_productions


def convert_json_countries(json_list):
    movie_countries = []
    for json_object in json_list:
        countries_id = [json_object['iso_3166_1'], json_object['name']]
        movie_countries.append(countries_id)

    return movie_countries


def convert_json_spoken_language(json_list):
    movie_spoken_langs = []
    for json_object in json_list:
        lang_id = [json_object['iso_639_1'], json_object['name']]
        movie_spoken_langs.append(lang_id)

    return movie_spoken_langs


def extract_first_item(data_list):
    firsts = []
    for element in data_list:
        firsts.append(element[1])

    return firsts


def extract_second_item(data_list):
    seconds = []
    for element in data_list:
        seconds.append(element[1])

    return seconds


def binarize_genre(data_list):
    binarizer = [0] * len(sorted_unique_genres)
    for element in data_list:
        element_index = sorted_unique_genres.index(element)
        binarizer[element_index] = 1

    return binarizer

'''

def binarize_keyword(data_list):
    binarizer = [0] * len(sorted_unique_keywords)
    for element in data_list:
        element_index = sorted_unique_keywords.index(element)
        binarizer[element_index] = 1

    return binarizer

'''


def binarize_companies(data_list):
    binarizer = [0] * len(sorted_unique_companies)
    for element in data_list:
        element_index = sorted_unique_companies.index(element)
        binarizer[element_index] = 1

    return binarizer


list_of_genres = []


def add_to_common_genre_list(data_list):
    for element in data_list:
        list_of_genres.append(element)


list_of_keywords = []


def add_to_common_keywords_list(data_list):
    for element in data_list:
        list_of_keywords.append(element)


list_of_companies = []


def add_to_common_companies_list(data_list):
    for element in data_list:
        list_of_companies.append(element)


pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

# read file
filename = "tmdb_5000_movies.csv"
data = pd.read_csv(filename, sep=',')

"""

Parsing JSON list to Python string list

"""

# clean genres
genres = data[['original_title', 'genres']]
genres = genres.copy()
genres['genres'] = genres['genres'].apply(json.loads)
genres['genres'] = genres['genres'].apply(convert_json_genres)
data['genres'] = genres['genres']

# clean keywords
keywords = data[['original_title', 'keywords']]
keywords = keywords.copy()
keywords['keywords'] = keywords['keywords'].apply(json.loads)
keywords['keywords'] = keywords['keywords'].apply(convert_json_keywords)
data['keywords'] = keywords['keywords']

# clean production companies
companies = data[['original_title', 'production_companies']]
companies = companies.copy()
companies['production_companies'] = companies['production_companies'].apply(json.loads)
companies['production_companies'] = companies['production_companies'].apply(convert_json_productions)
data['production_companies'] = companies['production_companies']

# clean production countries
countries = data[['original_title', 'production_countries']]
countries = countries.copy()
countries['production_countries'] = countries['production_countries'].apply(json.loads)
countries['production_countries'] = countries['production_countries'].apply(convert_json_countries)
data['production_countries'] = countries['production_countries']

# clean spoken language
languages = data[['original_title', 'spoken_languages']]
languages = languages.copy()
languages['spoken_languages'] = languages['spoken_languages'].apply(json.loads)
languages['spoken_languages'] = languages['spoken_languages'].apply(convert_json_spoken_language)
data['spoken_languages'] = languages['spoken_languages']

"""

**************** CLEANING AND ETL ****************

"""

data = data[data['genres'].apply(len).gt(0)]  # exclude observations with no genres recorded
# data = data[data['keywords'].apply(len).gt(0)]  # exclude observations with no keywords recorded
# data = data[data['production_companies'].apply(len).gt(0)]  # exclude observations with no production_companies recorded
data = data.copy().reset_index(drop=True)
# print(data)

''' EXTRACT GENRES '''

cleaned_genres = data[['original_title', 'genres']]
cleaned_genres = cleaned_genres.copy()
cleaned_genres['genres'].apply(add_to_common_genre_list)
genres_df = pd.DataFrame(list_of_genres, columns=['id', 'genre'])
# print(genres_df['id'].nunique())
# print(genres_df['genre'].nunique())
# print(genres_df['id'].unique())
# print(genres_df['genre'].unique())
sorted_unique_genres = np.sort(genres_df['genre'].unique()).tolist()
genres_collection = genres_df.groupby(['id', 'genre']).agg(count=('id', 'count')).reset_index()
# print(genres_collection)

cleaned_genres['genres'] = cleaned_genres['genres'].apply(extract_second_item)
cleaned_genres['genre_binarized'] = cleaned_genres['genres'].apply(binarize_genre)
new_df = pd.DataFrame(cleaned_genres['genre_binarized'].tolist(), columns=sorted_unique_genres)
cleaned_genres = pd.concat([cleaned_genres, new_df], axis=1)
# cleaned_genres[sorted_unique_genres] = cleaned_genres['genre_binarized'].apply(pd.Series)  # apply(pd.Series) works but slower
cleaned_genres = cleaned_genres.drop(['genres', 'genre_binarized'], axis=1)
cleaned_genres.to_csv(path_or_buf='genres_binarized.csv', index=False)
prepared_data = cleaned_genres
prepared_data['rating'] = data['vote_average']

grouped_data = prepared_data.groupby(sorted_unique_genres).agg(genres_avg=('rating', 'mean')).reset_index()
# grouped_data['group'] = np.arange(len(grouped_data))
prepared_data = prepared_data.merge(grouped_data, on=sorted_unique_genres)

"""

''' EXTRACT KEYWORDS '''

cleaned_keywords = data[['original_title', 'keywords']]
cleaned_keywords = cleaned_keywords.copy()
cleaned_keywords['keywords'].apply(add_to_common_keywords_list)
keywords_df = pd.DataFrame(list_of_keywords, columns=['id', 'keywords'])
# print(keywords_df['id'].nunique())
# print(keywords_df['keywords'].nunique())
# print(keywords_df['id'].unique())
# print(keywords_df['keywords'].unique())
sorted_unique_keywords = np.sort(keywords_df['keywords'].unique()).tolist()
keywords_collection = keywords_df.groupby(['id', 'keywords']).agg(count=('id', 'count')).reset_index()
# print(keywords_collection)

cleaned_keywords['keywords'] = cleaned_keywords['keywords'].apply(extract_second_item)
cleaned_keywords['keyword_binarized'] = cleaned_keywords['keywords'].apply(binarize_keyword)
new_df2 = pd.DataFrame(cleaned_keywords['keyword_binarized'].tolist(), columns=sorted_unique_keywords)
cleaned_keywords = pd.concat([cleaned_keywords, new_df2], axis=1)
cleaned_keywords = cleaned_keywords.drop(['keywords', 'keyword_binarized'], axis=1)
prepared_data = pd.concat([prepared_data, cleaned_keywords[cleaned_keywords.columns[1:]]], axis=1)

"""

"""

''' EXTRACT PRODUCTION COMPANIES '''

cleaned_companies = data[['original_title', 'production_companies']]
cleaned_companies = cleaned_companies.copy()
cleaned_companies['production_companies'].apply(add_to_common_companies_list)
companies_df = pd.DataFrame(list_of_companies, columns=['id', 'production_companies'])
# print(companies_df['id'].nunique())
# print(companies_df['production_companies'].nunique())
# print(companies_df['id'].unique())
# print(companies_df['production_companies'].unique())
sorted_unique_companies = np.sort(companies_df['production_companies'].unique()).tolist()
companies_collection = companies_df.groupby(['id', 'production_companies']).agg(count=('id', 'count')).reset_index()
# print(companies_collection)

cleaned_companies['production_companies'] = cleaned_companies['production_companies'].apply(extract_first_item)
cleaned_companies['companies_binarized'] = cleaned_companies['production_companies'].apply(binarize_companies)
new_df3 = pd.DataFrame(cleaned_companies['companies_binarized'].tolist(), columns=sorted_unique_companies)
cleaned_companies = pd.concat([cleaned_companies, new_df3], axis=1)
cleaned_companies = cleaned_companies.drop(['production_companies', 'companies_binarized'], axis=1)
prepared_data = pd.concat([prepared_data, cleaned_companies[cleaned_companies.columns[1:]]], axis=1)

"""

prepared_data['diff_from_avg'] = np.abs(prepared_data['rating'] - prepared_data['genres_avg'])
print(prepared_data)
# prepared_data.to_csv(path_or_buf='prepared_data.csv', index=False)

"""

**************** ANALYZING ****************

"""


plt.rcParams["figure.figsize"] = (18, 9)
# plt.hist(prepared_data['group'], bins=len(grouped_data))  # frequency of each group of genres
# plt.plot(prepared_data['group'], prepared_data['genres_avg'], 'bo', markersize=2)  # plot genre groups and their avg ratings
# plt.show()


''' Build a ML model to predict movie rating based on genres and keywords '''
col_names = sorted_unique_genres
col_names.append('diff_from_avg')
features = prepared_data[col_names]
ratings = prepared_data['rating']
X_train, X_valid, y_train, y_valid = train_test_split(features, ratings)

# print("KNeighbors Regressor:")
# model_kneighbors = KNeighborsRegressor(15)
# model_kneighbors.fit(X_train, y_train)
# print(model_kneighbors.score(X_train, y_train))
# print(model_kneighbors.score(X_valid, y_valid))
# print("")
#
# # This one gives low score of ~ 0.30 - 0.35
# # print("Random Forest Regressor:")
# # model_forest = RandomForestRegressor(1000, max_depth=5, min_samples_leaf=100, random_state=50)
# # model_forest.fit(X_train, y_train)
# # print(model_forest.score(X_train, y_train))
# # print(model_forest.score(X_valid, y_valid))
# # print("")
#
# # https://machinelearningmastery.com/configure-gradient-boosting-algorithm/
# print("Gradient Boost Regressor:")
# model_gradient_boost = GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_leaf=6)
# model_gradient_boost.fit(X_train, y_train)
# print(model_gradient_boost.score(X_train, y_train))
# print(model_gradient_boost.score(X_valid, y_valid))
# print("")
#
# print("Support Vector Machine Regressor:")
# model_svr = SVR(kernel='rbf')
# model_svr.fit(X_train, y_train)
# print(model_svr.score(X_train, y_train))
# print(model_svr.score(X_valid, y_valid))
# print("")
#
# print("Neural Network Regressor:")
# model_neural = make_pipeline(StandardScaler(),
#                              MLPRegressor(hidden_layer_sizes=(8, 6, 8), activation='logistic', solver='lbfgs',
#                                           max_iter=36000))
# model_neural.fit(X_train, y_train)
# print(model_neural.score(X_train, y_train))
# print(model_neural.score(X_valid, y_valid))
# print("")

# print("Voting Regressor:")
# model_voting = make_pipeline(StandardScaler(),
#                              VotingRegressor([
#                             ('gaussian', GaussianProcessRegressor()),
#                             ('neighbors', KNeighborsRegressor(50)),
#                             ('forest', RandomForestRegressor(n_estimators=100, min_samples_leaf=20)),
#                             ('svr', SVR()),
#                             ('neural', MLPRegressor(hidden_layer_sizes=(4, 5), activation='logistic', solver='lbfgs', max_iter=36000))
#                             ])
# )
#
# model_voting.fit(X_train, y_train)
# print(model_voting.score(X_train, y_train))
# print(model_voting.score(X_valid, y_valid))
# print("")
#
# results = prepared_data.copy()
# results['predict'] = model_voting.predict(results[col_names])
# results['diff_from_truth'] = np.abs(results['predict'] - results['rating'])
# print(results)

# data.to_csv(path_or_buf="output.csv", index=False)
