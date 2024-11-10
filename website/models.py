from . import db
from flask_login import UserMixin
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np
from website import model_name
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack
from sqlalchemy.types import LargeBinary, TypeDecorator
import pickle
import zlib
import faiss

model = SentenceTransformer(model_name)
faiss_index = None
vectorizer = None
tfidf_matrix = None

recipe_tag = db.Table('recipe_tag',
                    db.Column('recipe_id', db.Integer, db.ForeignKey('recipe.id')),
                    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id')),
                )
recipe_ingredient = db.Table('recipe_ingredient',
                    db.Column('recipe_id', db.Integer, db.ForeignKey('recipe.id')),
                    db.Column('ingredient_id', db.Integer, db.ForeignKey('ingredient.id'))
                )

class PickleType(TypeDecorator):
    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return pickle.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return pickle.loads(value)

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))    
    recipe_id = db.relationship('Recipe', backref='user')

@dataclass
class Recipe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150))
    cook_time = db.Column(db.Integer)
    steps = db.Column(db.Text)
    desc = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    tags = db.relationship('Tag', secondary=recipe_tag, backref='recipes')
    ingredients = db.relationship('Ingredient', secondary=recipe_ingredient, backref='recipes')
    images = db.relationship('Image', backref='recipe', lazy='dynamic')
    videos = db.relationship('Video', backref='recipe', lazy='dynamic')
    embedding = db.Column(PickleType)

@dataclass
class Tag(db.Model):
    id:int = db.Column(db.Integer, primary_key=True)
    name:str = db.Column(db.String(1000))

@dataclass
class Ingredient(db.Model):
    id:int = db.Column(db.Integer, primary_key=True)
    name:str = db.Column(db.String(1000))

@dataclass
class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    url = db.Column(db.String(1000))
    recipe_id = db.Column(db.Integer, db.ForeignKey('recipe.id'))

@dataclass
class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    url = db.Column(db.String(1000))
    length = db.Column(db.Integer)  # Duration in seconds
    recipe_id = db.Column(db.Integer, db.ForeignKey('recipe.id'))

@dataclass
class ModelStorage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), unique=True)
    data = db.Column(PickleType)
    updated_at = db.Column(db.DateTime, default=db.func.now(), onupdate=db.func.now())
    def set_data(self, obj):
        # Compress the pickled data
        pickled = pickle.dumps(obj)
        self.data = zlib.compress(pickled)

    def get_data(self):
        # Decompress the data
        if self.data:
            return pickle.loads(zlib.decompress(self.data))
        return None

# Function to prepare text for TF-IDF
def prepare_recipe_text(recipe):
    ingredients = ' '.join([i.name for i in recipe.ingredients])
    tags = ' '.join([t.name for t in recipe.tags])
    return f"{recipe.name} {ingredients} {recipe.desc} {tags}"

def set_faiss_index(index):
    global faiss_index
    faiss_index = index

def set_vectorizer_and_matrix(m_vectorizer, matrix):
    global vectorizer, tfidf_matrix
    vectorizer = m_vectorizer
    tfidf_matrix = matrix

def add_recipe_to_faiss(recipe):
    global faiss_index, vectorizer, tfidf_matrix
    
    # Update FAISS index
    embedding = np.array([recipe.embedding], dtype=np.float32)
    faiss_index.add(embedding)
    
    # Store updated FAISS index
    # faiss_storage = ModelStorage.query.filter_by(name='faiss_index').first()
    # if not faiss_storage:
    #     faiss_storage = ModelStorage(name='faiss_index')
    # faiss_storage.set_data(faiss_index)
    
    # Update TF-IDF
    recipe_text = prepare_recipe_text(recipe)
    new_tfidf = vectorizer.transform([recipe_text])
    tfidf_matrix = vstack([tfidf_matrix, new_tfidf])
    
    # Store updated TF-IDF matrix
    # tfidf_storage = ModelStorage.query.filter_by(name='tfidf_matrix').first()
    # if not tfidf_storage:
    #     tfidf_storage = ModelStorage(name='tfidf_matrix')
    # tfidf_storage.set_data(tfidf_matrix)
    
    print("Updated index and matrix")
    # db.session.add(faiss_storage)
    # db.session.add(tfidf_storage)
    # db.session.commit()

def remove_recipe_from_faiss(recipe):
    global faiss_index, tfidf_matrix
    
    # Remove the embedding from the FAISS index
    faiss_index.remove_ids(np.array([recipe.id - 1]))
    print(f"Removed recipe {recipe.id} from FAISS index.")
        
    # Store updated FAISS index in database
    # faiss_storage = ModelStorage.query.filter_by(name='faiss_index').first()
    # if not faiss_storage:
    #     faiss_storage = ModelStorage(name='faiss_index')
    # faiss_storage.set_data(faiss_index)

    #Removing from matrix
    row_to_remove = recipe.id - 1  # Assuming recipe IDs start from 1
    mask = np.ones(tfidf_matrix.shape[0], dtype=bool)
    mask[row_to_remove] = False
    tfidf_matrix = tfidf_matrix[mask]

    print(f"Removed recipe {recipe.id} from TF-IDF matrix.")
    print(f"Updated TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Store updated TF-IDF matrix in database
    # tfidf_storage = ModelStorage.query.filter_by(name='tfidf_matrix').first()
    # if not tfidf_storage:
    #     tfidf_storage = ModelStorage(name='tfidf_matrix')
    # tfidf_storage.set_data(tfidf_matrix)

    # Commit changes to database
    # db.session.add(faiss_storage)
    # db.session.add(tfidf_storage)
    # db.session.commit()
    # Updating the vectorizer takes a lot of time and resources. Thus, will only do it when absolutely neccesary
    
def combined_search_recipes(user_query, k_elements=100, semantic_threshold=0.1, tfidf_threshold=1, semantic_weight=0.4, tfidf_weight=0.6):
    global faiss_index, model, vectorizer, tfidf_matrix

    # Semantic Search (unchanged)
    query_embedding = model.encode(user_query)
    distances, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), k_elements)
    
    semantic_results = {}
    for distance, index in zip(distances[0], indices[0]):
        similarity = 1 / (1 + distance)  # Convert distance to similarity
        if similarity >= semantic_threshold:
            semantic_results[index + 1] = similarity

    # TF-IDF Search
    # Remove stop words and split the query
    query_terms = [term.lower() for term in user_query.split() if term.lower() not in ENGLISH_STOP_WORDS]
    query_vector = vectorizer.transform([user_query])

    # Get the feature names (words) from the vectorizer
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Find the indices of query terms in the feature names
    query_term_indices = [np.where(feature_names == term)[0][0] for term in query_terms if term in feature_names]
    
    # Find documents with matching keywords
    matching_docs = np.sum(tfidf_matrix[:, query_term_indices].toarray() > 0, axis=1)
    
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix[matching_docs]).flatten()

    tfidf_results = {}
    for i, (doc_match_count, similarity) in enumerate(zip(matching_docs, cosine_similarities)):

        if doc_match_count >= tfidf_threshold:
            recipe_id = i + 1
            # Score is directly proportional to the number of matching terms
            score = (doc_match_count + similarity)/ (len(query_terms) + 1)
            tfidf_results[recipe_id] = score
    
    # Combine results
    combined_results = {}
    all_recipe_ids = set(semantic_results.keys()) | set(tfidf_results.keys())
    
    for recipe_id in all_recipe_ids:
        semantic_score = semantic_results.get(recipe_id, 0)
        tfidf_score = tfidf_results.get(recipe_id, 0)
        combined_score = (semantic_weight * semantic_score) + (tfidf_weight * tfidf_score)
        combined_results[recipe_id] = combined_score

    # Sort results by score in descending order
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)

    # Return recipe IDs
    return [int(recipe_id) for recipe_id, _ in sorted_results]

def create_faiss_index(batch_size=1000):

    global faiss_index
    
    # Get total count of recipes with embeddings
    total_recipes = Recipe.query.filter(Recipe.embedding != None).count()
    # print(f"Total recipes with embeddings: {total_recipes}")

    # Adjust nlist based on total recipes
    nlist = min(256, max(1, total_recipes // 39))  # Ensure at least 39 points per centroid
    # print(f"Using {nlist} clusters for {total_recipes} recipes")

    try:
        # Collect training data
        training_data = []
        training_size = max(10000, 39 * nlist)  # Ensure we have at least 10,000 points or 39 * nlist, whichever is larger
        
        for i in range(0, training_size, batch_size):
            recipes = Recipe.query.filter(Recipe.embedding != None).with_entities(Recipe.embedding).offset(i).limit(batch_size).all()
            if not recipes:
                break
            embeddings = np.array([recipe.embedding for recipe in recipes], dtype=np.float32)
            training_data.append(embeddings)
        
        training_data = np.vstack(training_data)

        # Create and train the index
        embedding_dim = training_data.shape[1]
        m = 8  # number of subquantizers
        bits = 8  # bits per subquantizer

        quantizer = faiss.IndexFlatL2(embedding_dim)
        faiss_index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, m, bits)
        
        faiss_index.train(training_data)

        # Add all embeddings to the index
        for i in range(0, total_recipes, batch_size):
            
            recipes = Recipe.query.filter(Recipe.embedding != None).with_entities(Recipe.id, Recipe.embedding).offset(i).limit(batch_size).all()
            
            embeddings = np.array([recipe.embedding for recipe in recipes], dtype=np.float32)
            faiss_index.add(embeddings)
            
        
        print("FAISS index creation completed successfully.")

        return faiss_index
        
    except Exception as e:
        print(f"An error occurred during index creation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, []

def initialize_tfidvectorizer(all_recipes_ids_len):
    global vectorizer, tfidf_matrix
    recipe_texts = []
    for i in range(0, all_recipes_ids_len):    
        print(f"Formatting text for recipe {i+1}")
        recipe = Recipe.query.get(i+1)
        recipe_texts.append(prepare_recipe_text(recipe))

    # Initialize TfidfVectorizer with appropriate parameters
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

    # Fit the vectorizer to your text data
    tfidf_matrix = vectorizer.fit_transform(recipe_texts)

    return vectorizer, tfidf_matrix