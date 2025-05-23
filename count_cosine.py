#CountVectorizer + cosine similarity
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

movies = pd.read_csv("clusters_movies_with_tags.csv")
#пропускаем пустые тэги
movies = movies[movies["tag"].notna() & (movies["tag"].str.strip() != "")]
#удаление пробелов и приведение к нижнему регистру
movies["tags"] = movies["tag"].str.lower().str.strip()
#установка айди фильмов в индексы
movies.set_index("movieId", inplace=True)

def train_model():
    import os 
    os.makedirs("models", exist_ok=True)
    #векторизация и построение матрицы
    count_vec= CountVectorizer(token_pattern=r"(?u)\b\w[\w-]+\b") 
    count_matrix = count_vec.fit_transform(movies["tags"])
    joblib.dump((count_vec, count_matrix), "models/count_cosine.pkl")

def get_similar_movies(movie_id: int, top_n: int = 10):
    count_vec, count_matrix = joblib.load("models/count_cosine.pkl")
    #получение вектора фильма
    idx = movies.index.get_loc(movie_id)
    movie_vector = count_matrix[idx]

    #косинусное сходство между исходным вектором и векторами из матрицы
    cosine_similarities = cosine_similarity(movie_vector, count_matrix).flatten()

    #top_n наиболее похожих фильмов
    similar_indices = cosine_similarities.argsort()[::-1]
    similar_indices = [i for i in similar_indices if i != idx][:top_n]
    similar_movies = movies.iloc[similar_indices][["title", "tags"]].copy()
    similar_movies["similarity"] = cosine_similarities[similar_indices]
    similar_movies.reset_index(inplace=True)

    return similar_movies.to_dict(orient="records")

#пример: находим похожие на movieId=1
if __name__ == "__main__":
    #train_model()
    similar = get_similar_movies(movie_id=1, top_n=10)
    for s in similar:
        print(f"{s['movieId']}, {s['title']} — Сходство: {s['similarity']:.3f}")