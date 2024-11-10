from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import environ
from flask_login import LoginManager
from flask_caching import Cache
import faiss
import scipy.sparse as sp
import pickle
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

db = SQLAlchemy()
cache = Cache(config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300
})
model_name = 'paraphrase-mpnet-base-v2'

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = environ.get('SECRET_KEY', 'default_secret_key')
    
    # Use DATABASE_URL from Railway
    
    app.config['SQLALCHEMY_DATABASE_URI'] = environ.get('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    cache.init_app(app)


    from .views import views
    from .auth import auth
    
    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/auth')
    
    from .models import User 

    threading.Thread(target=async_create_database, args=(app,)).start()
    
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    return app

def async_create_database(app):
    with app.app_context():
        create_database(app, model_name)


def create_database(app, model_name):
    from .models import Recipe, ModelStorage, set_faiss_index, set_vectorizer_and_matrix, create_faiss_index, initialize_tfidvectorizer

    with app.app_context():
        
        recipe_count = Recipe.query.count()
        if recipe_count > 0:
            print("Database exists, checking for models...")
            cache.set('all_recipes_ids_len', recipe_count, timeout=0)

            # First check database for models
            try:
                faiss_storage = ModelStorage.query.filter_by(name='faiss_index').first()
                tfidf_storage = ModelStorage.query.filter_by(name='tfidf_matrix').first()
                vectorizer_storage = ModelStorage.query.filter_by(name='tfidf_vectorizer').first()
                print("Loading models from database...")
                set_faiss_index(faiss_storage.get_data())
                set_vectorizer_and_matrix(vectorizer_storage.get_data(), tfidf_storage.get_data())
            except:
                print("Models not found in database, loading from local files...")
                try:
                    print("creating faiss storage")
                    # Load from local files
                    faiss_index = create_faiss_index()
                    # Store in database
                    faiss_storage = ModelStorage(name='faiss_index')
                    faiss_storage.set_data(faiss_index)
                    print("Finished setting faiss storage")

                    print("creating vectorizer and matrix storage")
                    vectorizer, tfidf_matrix = initialize_tfidvectorizer(cache.get('all_recipes_ids_len'))

                    tfidf_storage = ModelStorage(name='tfidf_matrix')
                    tfidf_storage.set_data(tfidf_matrix)
                    
                    print("Finished setting tfidf storage")

                    vectorizer_storage = ModelStorage(name='tfidf_vectorizer')
                    vectorizer_storage.set_data(vectorizer)
                    print("Finished setting vectorizer storage")

                    set_faiss_index(faiss_index)
                    set_vectorizer_and_matrix(vectorizer, tfidf_matrix)
                    db.session.add(faiss_storage)
                    db.session.add(tfidf_storage)
                    db.session.add(vectorizer_storage)
                    db.session.commit()

                    print("Models stored in database")
                     # Set the models
                    
                except Exception as e:
                    raise Exception(f"Failed to load model files: {e}")


