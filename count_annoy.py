#CountVectorizer + Annoy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from annoy import AnnoyIndex
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

    #размерность векторов 
    vector_dim = count_matrix.shape[1]

    #создание annoy индекса
    annoy_index = AnnoyIndex(vector_dim, metric="angular") 
    for i in range(len(movies)):
        vector = count_matrix[i].toarray().flatten().astype("float32")
        annoy_index.add_item(i, vector)
    #деревья для индекса
    annoy_index.build(5)
    joblib.dump((count_vec, count_matrix), "models/count_annoy.pkl")
    annoy_index.save("models/count_annoy_index.ann")

def get_similar_movies(movie_id: int, top_n: int = 10):
    count_vec, count_matrix = joblib.load("models/count_annoy.pkl")
    vector_dim = count_matrix.shape[1]
    annoy_index = AnnoyIndex(vector_dim, metric="angular")
    annoy_index.load("models/count_annoy_index.ann")

    idx = movies.index.get_loc(movie_id)
    
    #получение top_n похожих фильмов
    similar_indices = annoy_index.get_nns_by_item(idx, top_n + 1, include_distances=True)
    indices = [i for i in similar_indices[0] if i != idx][:top_n]
    distances = similar_indices[1][1:top_n+1]
    similar_movies = movies.iloc[indices][["title", "tags"]].copy()
    
    #возврат похожести в шкале 0-1
    similar_movies["similarity"] = [1 - d / 2 for d in distances]
    similar_movies.reset_index(inplace=True)

    return similar_movies.to_dict(orient="records")

#пример: находим похожие на movieId=1
if __name__ == "__main__":
    #train_model()
    similar = get_similar_movies(movie_id=1, top_n=10)
    for s in similar:
        print(f"{s['movieId']}, {s['title']} — Сходство: {s['similarity']:.3f}")