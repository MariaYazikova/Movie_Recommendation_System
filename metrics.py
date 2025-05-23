import tfidf_cosine 
import tfidf_faiss 
import tfidf_annoy 
import hashing_cosine 
import hashing_faiss
import hashing_annoy
import count_cosine 
import count_faiss
import count_annoy 
import fasttext_tfidf_cosine 

import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib
import sys

movies = pd.read_csv("clusters_movies_with_tags.csv")
movies = movies[movies["tag"].notna() & (movies["tag"].str.strip() != "")]
movies["tags"] = movies["tag"].str.lower().str.strip()
movies.set_index("movieId", inplace=True)

models = [
    ("tfidf+cosine", tfidf_cosine),
    ("tfidf+faiss", tfidf_faiss),
    ("tfidf+annoy", tfidf_annoy),
    ("hashing+cosine", hashing_cosine),
    ("hashing+faiss", hashing_faiss),
    ("hashing+annoy", hashing_annoy),
    ("count+cosine", count_cosine),
    ("count+faiss", count_faiss),
    ("count+annoy", count_annoy),
    ("fasttext+tfidf+cosine", fasttext_tfidf_cosine),
]

#схожесть рекомендаций между собой
def intra_list_similarity(vectors):
    sim_matrix = cosine_similarity(vectors)
    tril_indices = np.tril_indices(sim_matrix.shape[0], k=-1)
    return sim_matrix[tril_indices].mean()

#схожесть рекомендаций на фильм-запрос
def mean_similarity_to_query(query_vector, result_vectors):
    sims = cosine_similarity(query_vector.reshape(1, -1), result_vectors)
    return sims.mean()

def get_vector_for_movie(matrix, movie_id):
    idx = movies.index.get_loc(movie_id)
    vec = matrix[idx]
    if hasattr(vec, "toarray"):
        vec = vec.toarray().flatten()
    elif isinstance(vec, np.matrix):
        vec = np.array(vec).flatten()
    return vec
    
with open('metris_output.txt', 'w', encoding='utf-8') as f:
    sys.stdout = f
    for name, module in models:
        print(f"\n{name}")
        start = time.time()
        module.train_model()
        end = time.time()
        print(f"обучение: {end-start:.4f} сек")
        
        pkl_path = f"models/{name.replace('+','_')}.pkl"
        if name == "fasttext+tfidf+cosine":
            model, tfidf, movies = joblib.load(pkl_path)
            matrix = np.vstack(movies["vector"])
        else:
            vec, matrix = joblib.load(pkl_path)

        #1 - Toy Story, 5618 - Spirited Away, 1721 - Titanic
        movie_ids = [1, 5618, 1721]
        total_time = 0
        ils_scores = []
        mean_sim_scores = []
        
        for movie_id in movie_ids:
            start = time.time()
            results = module.get_similar_movies(movie_id, top_n=10)
            end = time.time()
            res = end - start
            total_time+=res
            print(f"поиск похожих для фильма {movie_id}: {res:.4f} сек")
            
            query_vector = get_vector_for_movie(matrix, movie_id)
            result_vectors = np.vstack([get_vector_for_movie(matrix, r["movieId"]) for r in results])  
            
            ils = intra_list_similarity(result_vectors)
            mean_sim = mean_similarity_to_query(query_vector, result_vectors)
            
            ils_scores.append(ils)
            mean_sim_scores.append(mean_sim)
            
            print(f"ILS: {ils:.4f}")
            print(f"MeanSim: {mean_sim:.4f}")
            
        average_time = total_time/(len(movie_ids))
        average_ils = np.mean(ils_scores)
        average_mean_sim = np.mean(mean_sim_scores)
        print(f"среднее время поиска: {average_time:.4f} сек")
        print(f"средний ILS: {average_ils:.4f}")
        print(f"средний MeanSim: {average_mean_sim:.4f}")
    sys.stdout = sys.__stdout__
