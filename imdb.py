###################################
##############IMDB SORTING#################
###################################


import pandas as pd
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import math
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None

# read csv
df = pd.read_csv("datasets/movies_metadata.csv")
df = df[["title", "vote_average", "vote_count"]]
print(df.shape)
df.head(10)


##data analysis
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Types #####################")
    print(dataframe.info())
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### MissingValues #######################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).T)


check_df(df)

##sorting by vote average
df.sort_values("vote_average", ascending=False).head()
df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T

# vote count standatation

df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)). \
    fit(df[["vote_count"]]). \
    transform(df[["vote_count"]])

# vote_count_score * vote_average

df["average_count_score"] = df["vote_count_score"] * df["vote_average"]

df.sort_values("average_count_score", ascending=False).head(20)

##IMDB WEIGHTED RATING

# weighted_rating = (v /(v+M) * r)+(M/(v+M) * C)
# r= vote average
# M= minimum votes required to be  listed  in the top 250
# v= vote count
# C= the mean across the whole report (current 7.0)

M = 2500
C = df["vote_average"].mean()


def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)


# our sorting
df.sort_values("average_count_score", ascending=False).head(20)

# IMDB formulation sorting
df["weighted_rating"] = weighted_rating(df["vote_average"], df["vote_count"], M, C)

df.sort_values("weighted_rating", ascending=False).head(10)

top_10_ratings = df.sort_values(by="weighted_rating", ascending=False)[0:10]
fig = px.scatter(top_10_ratings, y='title', x='weighted_rating',  title="Top 10 High Rated Movies")

fig.show()

# The top 10 'Title's with the highest IMDB Ratings and visualization.

df.groupby("title").agg({"weighted_rating": "max"}).sort_values(by="weighted_rating", ascending=False)[0:10]
score = df.groupby("title").agg({"weighted_rating": "max"}).sort_values(by="weighted_rating", ascending=False)[
                0:10].reset_index()

print (df.groupby("title").agg({"weighted_rating": "max"}).sort_values(by="weighted_rating", ascending=False)[0:10])
sns.lineplot(y=score["title"], x=score["weighted_rating"])
plt.show()



# Bayesian Avarage Rating Score
def bayesian_average_rating(n, confidence=0.05):
    # n: starst count ,frecuency
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
        score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
        return (score)

# read score distribution csv file
df = pd.read_csv("datasets/imdb_ratings.csv")
df = df.iloc[0:, 1:]

#
df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five"
    , "six", "seven", "nine", "ten"]]), axis=1)

df.sort_values("bar_score", ascending=False).head(10)


###################################################
# SORTING REWIEVS
###################################################

# UP-DOWN DIF SCORE = (up ratings) -(down ratings)

# Review 1:600 up 400 down total 1000
# Review 2:5500 up 4500 down total 1000

def score_up_down_diff(up, down):
    return up - down


score_up_down_diff(600, 400)
score_up_down_diff(5500, 4500)


# Score= Average rating = (up ratings)/(all ratings)

def score_average_rating(up, down):
    if (up + down) == 0:
        return 0
    return up / (up + down)


# daha mantıklı değerler geldi
score_average_rating(600, 400)
score_average_rating(5500, 4500)


###################################################
# WİLSON  LOWER BOUND SCORE:yorum sıralamak için kullanılır.
###################################################
def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


# daha mantıklı değerler geldi
wilson_lower_bound(600, 400)
wilson_lower_bound(5500, 4500)
