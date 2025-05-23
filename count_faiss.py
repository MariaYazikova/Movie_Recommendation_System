#CountVectorizer + Faiss
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import faiss
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
    count_vec = CountVectorizer(token_pattern=r"(?u)\b\w[\w-]+\b") 
    count_matrix = count_vec.fit_transform(movies["tags"])
    count_matrix_dense = count_matrix.toarray().astype("float32")

    #создание faiss индекса
    index = faiss.IndexFlatIP(count_matrix_dense.shape[1])
    faiss.normalize_L2(count_matrix_dense)
    index.add(count_matrix_dense)
    
    faiss.write_index(index, "models/count_faiss_index.index")
    joblib.dump((count_vec, count_matrix_dense), "models/count_faiss.pkl")

def get_similar_movies(movie_id: int, top_n: int = 10):
    count_vec, count_matrix_dense = joblib.load("models/count_faiss.pkl")
    index = faiss.read_index("models/count_faiss_index.index")
    
    #получение вектора фильма
    idx = movies.index.get_loc(movie_id)
    movie_vector = count_matrix_dense[idx].reshape(1, -1)

    #top_n наиболее похожих фильмов
    D, I = index.search(movie_vector, top_n + 1)
    similar_indices = [i for i in I[0] if i != idx][:top_n]
    similar_movies = movies.iloc[similar_indices][["title", "tags"]].copy()
    similar_movies["similarity"] = D[0][1:top_n+1]
    similar_movies.reset_index(inplace=True)

    return similar_movies.to_dict(orient="records")

#пример: находим похожие на movieId=1
if __name__ == "__main__":
    #train_model()
    similar = get_similar_movies(movie_id=1, top_n=10)
    for s in similar:
        print(f"{s['movieId']}, {s['title']} — Сходство: {s['similarity']:.3f}")