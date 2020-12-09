import json
import pandas as pd
import re
import numpy as np

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

# read file
filename = "tmdb_5000_movies.csv"
data = pd.read_csv(filename, sep=',')


"""

Notice that each element in the list of JSON object has only 2 attributes: the first one is the unique key, which is
either 'id' or 'iso_', and the second one is 'name'. Therefore, we parse JSON list to Python string list for later use.

"""

# clean genres
genres = data[['original_title', 'genres']]
genres = genres.copy()
genres['genres'] = genres['genres'].apply(json.loads)
genres['genres'] = genres['genres'].apply(lambda x: [[i['id'], i['name']] for i in x] if isinstance(x, list) else [])
data['genres'] = genres['genres']
list_of_genres = []
genres['genres'].apply(lambda x: [list_of_genres.append(i) for i in x])
genres_df = pd.DataFrame(list_of_genres, columns=['genre_id', 'genre'])
# print(genres_df['genre_id'].nunique())
# print(genres_df['genre'].nunique())
# print(genres_df['genre_id'].unique())
# print(genres_df['genre'].unique())
genres_collection = genres_df.groupby(['genre_id', 'genre']).agg(count=('genre_id', 'count')).sort_values(by=['genre']).reset_index()
genres_collection.to_csv(path_or_buf='genres_collection.csv', index=False)

# clean keywords
keywords = data[['original_title', 'keywords']]
keywords = keywords.copy()
keywords['keywords'] = keywords['keywords'].apply(json.loads)
keywords['keywords'] = keywords['keywords'].apply(lambda x: [[i['id'], i['name']] for i in x] if isinstance(x, list) else [])
data['keywords'] = keywords['keywords']
list_of_keywords = []
keywords['keywords'].apply(lambda x: [list_of_keywords.append(i) for i in x])
keywords_df = pd.DataFrame(list_of_keywords, columns=['keyword_id', 'keyword'])
# print(keywords_df['keyword_id'].nunique())
# print(keywords_df['keyword'].nunique())
# print(keywords_df['keyword_id'].unique())
# print(keywords_df['keyword'].unique())
keywords_collection = keywords_df.groupby(['keyword_id', 'keyword']).agg(count=('keyword_id', 'count')).sort_values(by=['keyword']).reset_index()
keywords_collection.to_csv(path_or_buf='keywords_collection.csv', index=False)

# clean production companies
companies = data[['original_title', 'production_companies']]
companies = companies.copy()
companies['production_companies'] = companies['production_companies'].apply(json.loads)
companies['production_companies'] = companies['production_companies'].apply(lambda x: [[i['id'], i['name']] for i in x] if isinstance(x, list) else [])
data['production_companies'] = companies['production_companies']
list_of_companies = []
companies['production_companies'].apply(lambda x: [list_of_companies.append(i) for i in x])
companies_df = pd.DataFrame(list_of_companies, columns=['company_id', 'company'])
# print(companies_df['company_id'].nunique())
# print(companies_df['company'].nunique())
# print(companies_df['company_id'].unique())
# print(companies_df['company'].unique())
companies_collection = companies_df.groupby(['company_id', 'company']).agg(count=('company_id', 'count')).sort_values(by=['company']).reset_index()
''' there are 29 companies with the same name but different companies; will keep the data based on company_id for now '''
# diff = companies_collection.groupby('company').agg(num_comp=('company_id', 'count')).reset_index()
# diff = diff[diff['num_comp'] > 1]
companies_collection.to_csv(path_or_buf='companies_collection.csv', index=False)

# clean production countries
countries = data[['original_title', 'production_countries']]
countries = countries.copy()
countries['production_countries'] = countries['production_countries'].apply(json.loads)
countries['production_countries'] = countries['production_countries'].apply(lambda x: [[i['iso_3166_1'], i['name']] for i in x] if isinstance(x, list) else [])
data['production_countries'] = countries['production_countries']
list_of_countries = []
countries['production_countries'].apply(lambda x: [list_of_countries.append(i) for i in x])
countries_df = pd.DataFrame(list_of_countries, columns=['country_id', 'country'])
# print(countries_df['country_id'].nunique())
# print(countries_df['country'].nunique())
# print(countries_df['country_id'].unique())
# print(countries_df['country'].unique())
countries_collection = countries_df.groupby(['country_id', 'country']).agg(count=('country_id', 'count')).sort_values(by=['country']).reset_index()
countries_collection.to_csv(path_or_buf='countries_collection.csv', index=False)

# clean spoken language
languages = data[['original_title', 'spoken_languages']]
languages = languages.copy()
languages['spoken_languages'] = languages['spoken_languages'].apply(json.loads)
languages['spoken_languages'] = languages['spoken_languages'].apply(lambda x: [[i['iso_639_1'], i['name']] for i in x] if isinstance(x, list) else [])
data['spoken_languages'] = languages['spoken_languages']
list_of_languages = []
languages['spoken_languages'].apply(lambda x: [list_of_languages.append(i) for i in x])
languages_df = pd.DataFrame(list_of_languages, columns=['language_id', 'language'])
# print(languages_df['language_id'].nunique())
# print(languages_df['language'].nunique())
# print(languages_df['language_id'].unique())
# print(languages_df['language'].unique())
''' there are 26 rows with empty cells for language '''
languages_df = languages_df[languages_df['language'].apply(len).gt(0)]  # exclude rows with no languages recorded
languages_collection = languages_df.groupby(['language_id', 'language']).agg(count=('language_id', 'count')).sort_values(by=['language']).reset_index()
languages_collection.to_csv(path_or_buf='languages_collection.csv', index=False)

# drop unnecessary columns that we do not use before output into CSV file to reduce the file size
data = data.drop(columns=['homepage', 'status'])
data.to_csv(path_or_buf='movies.csv', index=False)
