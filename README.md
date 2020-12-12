# CMPT 353 - Project
## Brian Lee & Ha Thu Nguyen

The movie industry is full of intriguing mysteries and presents an interesting opportunity to apply data science. Therefore, the purpose of this project was to analyze the relationship between the success of a movie and a movie’s plot.

1.	What is the current trend of movie genres in the industry?
2.	Which genres tend to be more highly rated by viewers? Is there a relationship between movie genres and movie rating?
3. 	Based on a movie’s plot overview, which genre should this movie be classified to?
4.	What is the likelihood that a movie will return ‘high revenue’ before the movie is released?
5.	Is it possible to build a movie recommendation system to determine a target audience?

## Required Libraries
- numpy, pandas, matplotlib, NLTK, json, string, sklearn

## Data Source 
- tmdb_5000_movies.csv

The CSV file contains data on 4800 different movies and originates from The Movie Database (TMDb)

## Commands
### complete_script.ipynb
The jupyter notebook - complete_script.ipynb - shows an overview of the data analysis and steps to preprocess data. It includes 

a) Genre classification (multi-label) - preliminary model 

b) Predictive model to predict the likelihood that a movie will return high revenue

c) Movie recommendation system based on cosine similarity scores

command: 'jupyter-notebook complete_script.ipynb'.

### predict_movie_genres.py

predict_movie_genres.py shows the details of the genre classification and contains the 'best' model, as well as model comparison

command: 'python3 predict_movie_genres.py' 

### Other Python Scripts

The other python scripts (analyze_movies.py, rating_regressor.py and parse_data.py) show the different approaches and steps taken for this project. These results were unused in our final report.

### Report

The summary of our findings and conclusions are found in the Report.

