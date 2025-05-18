Step 1: Data Merging
Combine the movies.csv and credits.csv datasets using a common key (id and movie_id).

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
df = movies.merge(credits, left_on='id', right_on='movie_id')

 Step 2: Feature Extraction
Extract key metadata:

Genres: Movie categories

Cast: Top 3 actors

Director: Extracted from the crew data

def get_director(crew_list):
    for person in crew_list:
        if person.get('job') == 'Director':
            return person['name']
    return ''

# Apply on respective columns
df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)])
df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])
df['crew'] = df['crew'].apply(ast.literal_eval)
df['director'] = df['crew'].apply(get_director)

Step 4: Vectorization (TF-IDF)
Convert the textual metadata into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency).

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
vectors = tfidf.fit_transform(df['combined_features'])

Step 5: Similarity Calculation
Compute cosine similarity between all movie vectors to measure how similar each movie is to the others.

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

 Step 6: Recommendation Function
Fetch the most similar movies for a given input movie.

def recommend(movie_title):
    idx = df[df['title'] == movie_title].index[0]
    distances = list(enumerate(similarity[idx]))
    movies_sorted = sorted(distances, reverse=True, key=lambda x: x[1])[1:6]
    for i in movies_sorted:
        print(df.iloc[i[0]]['title'])
