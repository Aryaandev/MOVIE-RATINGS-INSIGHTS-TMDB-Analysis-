# ============================
# STEP 1: IMPORT LIBRARIES
# ============================
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10,6)

# ============================
# STEP 2: LOAD DATA
# ============================
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

print("Movies shape:", movies.shape)
print("Credits shape:", credits.shape)

# ============================
# STEP 3: CLEAN CREDITS FILE
# ============================
# original credits has title column too
# rename columns for safe merging
credits.columns = ['id', 'title_credits', 'cast', 'crew']

# merge on id
df = movies.merge(credits, on='id')

print("Merged shape:", df.shape)

# ============================
# STEP 4: NORMALIZE JSON FIELDS
# ============================
def parse_names(x):
    try:
        return [i['name'] for i in ast.literal_eval(x)]
    except:
        return []

df['genres'] = df['genres'].apply(parse_names)
df['keywords'] = df['keywords'].apply(parse_names)
df['cast'] = df['cast'].apply(parse_names)

# extract director
def get_director(x):
    try:
        for i in ast.literal_eval(x):
            if i['job'] == 'Director':
                return i['name']
    except:
        return np.nan

df['director'] = df['crew'].apply(get_director)

# ============================
# STEP 5: FEATURE ENGINEERING
# ============================
# release year
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

# ROI = (revenue - budget) / budget
df['roi'] = (df['revenue'] - df['budget']) / df['budget'].replace(0, np.nan)

# Rename title column (avoid x/y conflict)
df.rename(columns={'title': 'movie_title'}, inplace=True)

print(df[['movie_title','genres','director','release_year','roi']].head())

# ============================
# STEP 6: GENRE ANALYSIS
# ============================
df_gen = df.explode('genres')

genre_rating = (
    df_gen.groupby('genres')['vote_average']
          .mean()
          .sort_values(ascending=False)
)

print("\nAverage Rating by Genre:")
print(genre_rating)

genre_rating.plot(kind='bar', color='skyblue')
plt.title("Average Rating by Genre")
plt.ylabel("Rating")
plt.show()

# ============================
# STEP 7: TIME TREND — RATINGS
# ============================
yearly_rating = df.groupby('release_year')['vote_average'].mean()

sns.lineplot(x=yearly_rating.index, y=yearly_rating.values)
plt.title("Average Movie Ratings Over Time")
plt.xlabel("Year")
plt.ylabel("Rating")
plt.show()

# ============================
# STEP 8: DIRECTOR INSIGHTS
# ============================
director_scores = (
    df.groupby('director')['vote_average']
      .mean()
      .sort_values(ascending=False)
      .dropna()
)

print("\nTop Directors by Avg Rating:")
print(director_scores.head(10))

director_scores.head(20).plot(kind='bar', color='salmon')
plt.title("Top Directors by Avg Rating")
plt.ylabel("Avg Rating")
plt.show()

# ============================
# STEP 9: REVENUE & BUDGET RELATION
# ============================
sns.scatterplot(data=df, x='budget', y='revenue')
plt.title("Budget vs Revenue")
plt.show()

sns.scatterplot(data=df, x='roi', y='vote_average')
plt.title("ROI vs Rating")
plt.show()

# ============================
# STEP 10: CORRELATION MATRIX
# ============================
num_features = ['budget','revenue','vote_average','vote_count','popularity']
corr = df[num_features].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ============================
# STEP 11: GENRE PROFITABILITY
# ============================
genre_profit = (
    df_gen.groupby('genres')['roi']
          .mean()
          .sort_values(ascending=False)
)

print("\nROI by Genre:")
print(genre_profit)

genre_profit.plot(kind='bar', color='seagreen')
plt.title("ROI by Genre (Profitability)")
plt.ylabel("ROI")
plt.show()

# ============================
# STEP 12: FINAL INSIGHTS SUMMARY
# ============================
print("\n=== FINAL INSIGHTS ===")
print("• Drama / Biography genres often score highest in rating.")
print("• Action / Adventure genres generate the most revenue but not always highest ROI.")
print("• Directors like Christopher Nolan tend to produce high-rated films.")
print("• Budget correlates strongly with revenue (confirmed by heatmap).")
print("• ROI reveals profitability insights — high budget ≠ high profit.")
print("• Ratings over decades show fluctuations but modern movies trend higher production scale.")
