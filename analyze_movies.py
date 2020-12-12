import ast
import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MultiLabelBinarizer

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
plt.rcParams["figure.figsize"] = (23, 20)
sns.set_theme(style="darkgrid")


def map_rating_rank(value, high, low):
    if value >= high:
        return "High"
    elif value <= low:
        return "Low"
    else:
        return "Medium"


# read file
filename = "movies.csv"
data = pd.read_csv(filename, sep=',', converters={'genres': ast.literal_eval}, parse_dates=['release_date'])

# ETL
data = data[data['genres'].apply(len).gt(0)]  # exclude observations with no genres recorded
data = data[data['overview'].notna()]  # exclude observations with no overview recorded
data = data[data['vote_average'].notna()]  # exclude observations with no vote_average recorded
data = (data[data['vote_average'] <= 10])  # exclude observations with invalid vote > 10 (since vote is out of 10)
data = data[data['release_date'].notna()]  # exclude observations with no released date recorded
data = data[data['vote_count'] != 0]  # exclude observations with no vote counts recorded
data = data.copy().reset_index(drop=True)


''' NUMBER OF MOVIES PER GENRES '''

movies = data[['original_title', 'genres', 'vote_average', 'release_date']]
movies = movies.copy()
genres = pd.read_csv("genres_collection.csv", sep=',')
unique_genres = genres['genre'].tolist()
movies['genres'] = movies['genres'].apply(lambda x: [i[1] for i in x])
mlb = MultiLabelBinarizer()
genres_binarized = pd.DataFrame(mlb.fit_transform(movies['genres']), columns=mlb.classes_, index=movies.index)
movies = movies.join(genres_binarized)
movies = movies.drop(['genres'], axis=1)
num_genres = pd.DataFrame(movies[unique_genres].sum()).reset_index()
num_genres.columns = ['genres', 'count']
num_genres = num_genres.sort_values(by='count', ascending=False)
# print(num_genres)

# genres_count = sns.barplot(data=num_genres, x='genres', y='count', palette="hls")
# genres_count.set_xlabel("Genres", fontsize=20)
# genres_count.set_ylabel("Number of movies", fontsize=20)
# genres_count.set_xticklabels(genres_count.get_xticklabels(), rotation=45, fontsize=16)
# genres_count.set_title("Distribution Of Movies By Genres", fontsize=25)
# plt.savefig("genres_count.png")


''' AVERAGE VOTING PER GENRE '''

vote_melt = pd.melt(movies, id_vars=['original_title', 'vote_average'], value_vars=unique_genres, var_name='genre')
vote_melt_valid = vote_melt[vote_melt['value'] == 1]
vote_melt_valid = vote_melt_valid.drop('value', axis=1)  # number of rows should be equal to num_genres.sum()
# print(num_genres.sum())
vote_melt_valid['vote_sq'] = np.exp(vote_melt_valid['vote_average'])
# print(vote_melt_valid)

# normality = [stats.normaltest(vote_melt_valid[vote_melt_valid['genre'] == x]['vote_sq']).pvalue for x in unique_genres]
# https://stats.stackexchange.com/questions/56971/alternative-to-one-way-anova-unequal-variance
# variance = [stats.levene(vote_melt_valid[vote_melt_valid['genre'] == x]['vote_sq'] for x in unique_genres).pvalue]
# variance = pg.homoscedasticity([vote_melt_valid[vote_melt_valid['genre'] == x]['vote_sq'].to_numpy() for x in unique_genres])

# Plotting for each genre shows the data looks close to normal, although it fails the normality test. We choose to carry on...

# genres_rating = sns.boxplot(data=vote_melt_valid, x='genre', y='vote_average', palette="Spectral")
# genres_rating.set_xlabel("Genres", fontsize=20)
# genres_rating.set_ylabel("Average rating of genre", fontsize=20)
# genres_rating.set_title("Box and Whisker Plot of Movie Rating Distribution By Genres", fontsize=25)
# plt.savefig("genres_avg_rating.png")


''' TREND OF GENRES '''

genres_by_year = movies
genres_by_year = genres_by_year.copy()
genres_by_year['year'] = pd.DatetimeIndex(genres_by_year['release_date']).year
genres_by_year = genres_by_year.drop(['original_title', 'vote_average', 'release_date'], axis=1)
grouped_data = genres_by_year.groupby('year').agg('sum').reset_index()
year_melt = pd.melt(grouped_data, id_vars=['year'], value_vars=unique_genres, var_name='genre')

# genres_trend = sns.lineplot(data=year_melt, x='year', y='value', hue='genre', palette='icefire')
# genres_trend.set_xlabel("Year", fontsize=20)
# genres_trend.set_ylabel("Number of Movies", fontsize=20)
# genres_trend.set_title("Trend Of Movie Genres Over 100-Year Period Between 1916 And 2017", fontsize=25)
# plt.savefig("genres_trend.png")


''' HIGHLY RATED GENRES '''

votes = movies.groupby('vote_average').agg(movie_count=('original_title', 'count')).reset_index()

# vote_distribution = sns.barplot(data=votes, x='vote_average', y='movie_count', palette='rocket')
# vote_distribution.set_xlabel("Movie Vote Average", fontsize=20)
# vote_distribution.set_ylabel("Frequency", fontsize=20)
# vote_distribution.set_title("Distribution of movie vote average", fontsize=25)
# plt.savefig("vote_distribution.png")

print("P-value for normality test of voting distribution: ", end="")
print(stats.normaltest(votes['vote_average']).pvalue)  # pvalue = 0.45 > 0.05 --> proceed as if vote_average has normal distribution

'''
A movie with rating in the 75th percentile is considered high rated.
A movie with rating below the 25th percentile is considered low rated.
A movie with a rating in between is medium.
 '''
quantile75 = votes['vote_average'].quantile(0.75)
quantile25 = votes['vote_average'].quantile(0.25)
print("The 75th percentile is " + str(quantile75))
print("The 25th percentile is " + str(quantile25))
movies_rank = movies
movies_rank = movies_rank.copy()
movies_rank['rank'] = movies_rank['vote_average'].apply(map_rating_rank, args=(quantile75, quantile25))
movies_rank = movies_rank.drop(['original_title', 'vote_average', 'release_date'], axis=1)

ranks = movies_rank.groupby('rank').agg('sum').reset_index()
# print(ranks)
# More than 80% of the contigency table have more than 5 observations in each cells, so we can use Chi2 Test
contingency_table = ranks[unique_genres].to_numpy()
chi2, chi2_pvalue, dof, expected = stats.chi2_contingency(contingency_table, correction=True)
print("P-value for Chi-square Test is ", end="")
print(chi2_pvalue)  # pvalue = 5*10^(-73) --> very small --> The genres do affect whether a movie is highly rated or lowly rated
