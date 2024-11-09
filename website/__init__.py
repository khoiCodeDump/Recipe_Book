from flask import Flask, current_app
from flask_sqlalchemy import SQLAlchemy
from os import path, environ
from flask_login import LoginManager
import ast
import pandas as pd
from flask_caching import Cache
import faiss
import numpy as np
import scipy.sparse as sp
import pickle
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

db = SQLAlchemy()
DB_NAME = "database"
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

def create_weighted_embedding(model, recipe, weights):
    ingredients_text = ', '.join([ingredient.name for ingredient in recipe.ingredients])
    tags_text = ', '.join([tag.name.replace('-', ' ') for tag in recipe.tags])
    numbered_steps = " ".join([f"{i+1}. {step}" for i, step in enumerate(recipe.steps.split("|"))])

    text_data = (
        f"The recipe name is {recipe.name}",
        f"The recipe takes {recipe.cook_time} minutes to cook",
        f"To cook the recipe, the following ingredients are required, separated by commas: {ingredients_text}.",
        f"The recipe has the following associated tags, separated by commas: {tags_text}.",
        f"The description of the recipe is: {recipe.desc}. ",
        f"Here are the instructions to cook the recipe: {numbered_steps}."
    )
    embeddings = model.encode(text_data)
    weighted_embedding = np.average(embeddings, axis=0, weights=weights)
    return weighted_embedding

def update_recipes_embeddings(model):
    from .models import Recipe

    print("Updating recipe embeddings...")
    recipes_len = Recipe.query.count()
    batch_size = 100
    weights = [1.2, 1.1, 1.5, 1.4, 1.1, 1.2]
    for i in range(0, recipes_len, batch_size):            
        # Query recipes in the current batch
        recipes = Recipe.query.offset(i).limit(batch_size).all()
    
        for recipe in recipes:
            print(f"Updating recipe {recipe.id}")
            # Generate new embedding for each recipe
            new_embedding = create_weighted_embedding(model, recipe, weights)
            
            # Update the recipe's embedding
            recipe.embedding = new_embedding.tolist()

        print(f"Committed recipes {i+1} - {min(i + batch_size, recipes_len)}")
        db.session.commit()
    print("Recipe embeddings updated successfully.")


def async_create_database(app):
    with app.app_context():
        create_database(app, model_name)

def create_database(app, model_name):
    from .models import Recipe, Tag, Ingredient, create_faiss_index, set_faiss_index, initialize_tfidvectorizer, set_tfidvectorizer
    from sentence_transformers import SentenceTransformer

    with app.app_context():
        model = SentenceTransformer(model_name)

        # Check if the Recipe table exists and is empty
        try:
            recipe_count = db.session.query(Recipe).count()
            ingredient_count = db.session.query(Ingredient).count()
            tag_count = db.session.query(Tag).count()
            
            if recipe_count > 0 and ingredient_count > 0 and tag_count > 0:
                table_exists = True
        except:
            table_exists = False
            
        if not table_exists:
            tags = {}
            ingredients = {}
            db.create_all()
            raw_recipes = pd.read_csv('archive/RAW_recipes.csv')
    
            for index, row in raw_recipes.iterrows():
                ingredients_list = ast.literal_eval(row["ingredients"])
                m_steps = ast.literal_eval(row["steps"])
                tags_list = ast.literal_eval(row["tags"])
    
                steps_list = '|'.join(m_steps)
                
                recipe = Recipe(name = row["name"], cook_time = int(row["minutes"]),steps = steps_list, desc = row["description"])
                
                db_tag_list = []
                db_ingredient_list = []
                
                for ingredient in ingredients_list:
                    if not ingredient:
                        continue
                    if ingredient not in ingredients:
                        db_ingredient = Ingredient(name=ingredient)
                        ingredients[ingredient] = db_ingredient
                        db_ingredient_list = db_ingredient_list + [db_ingredient]
                    else:
                        db_ingredient = ingredients[ingredient]

                    recipe.ingredients.append(db_ingredient)
                    
                for tag in tags_list:
                    if not tag:
                        continue
                    if tag not in tags:
                        db_tag = Tag(name=tag)
                        tags[tag] = db_tag
                        db_tag_list = db_tag_list + [db_tag]
                    else:
                        db_tag = tags[tag]
                        
                    recipe.tags.append(db_tag)

                
                ingredients_text = ', '.join(ingredients_list)
                tags_text = ', '.join(tags_list)

                numbered_steps = " ".join([f"{i+1}. {step}" for i, step in enumerate(m_steps)])

                text_data = (
                    f"The recipe name is {row['name']}",
                    f"The recipe takes {row['minutes']} minutes to cook",
                    f"To cook the recipe, the following ingredients are required, separated by commas: {ingredients_text}."
                    f"The recipe has the following associated tags, separated by commas: {tags_text}."
                    f"The description of the recipe is: {row['description']}. "
                    f"Here are the instructions to cook the recipe: {numbered_steps}."
                )
                
                # Generate embedding for the recipe
                embedding = model.encode(text_data)
                
                # Update the recipe with the generated embedding
                recipe.embedding = embedding.tolist()


                db.session.add_all(db_tag_list)
                db.session.add_all(db_ingredient_list)
                db.session.add(recipe)
                db.session.commit()
            
                print(f"Commited {recipe.id} to database")

            print('Created Database!')
        

        if not path.exists(f'recipe_index_{model_name}.faiss'):
            update_recipes_embeddings(model)
            m_faiss_index = create_faiss_index()
            faiss.write_index(m_faiss_index, f'recipe_index_{model_name}.faiss')
        else:
            set_faiss_index(faiss.read_index(f'recipe_index_{model_name}.faiss'))


        cache.set('all_recipes_ids_len', Recipe.query.count(), timeout=0)

        if not (path.exists('tfidf_matrix.npz') or path.exists('tfidf_vectorizer.pkl') ):
            print("Vectorizer and matrix do not exist")
            vectorizer, tfidf_matrix = initialize_tfidvectorizer(cache.get('all_recipes_ids_len'))
            with open('tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(vectorizer, f)
            sp.save_npz('tfidf_matrix.npz', tfidf_matrix)
        else:
            print("Vectorizer and matrix both exist")
            with open('tfidf_vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)

            # Load the tfidf matrix
            tfidf_matrix = sp.load_npz('tfidf_matrix.npz')

            set_tfidvectorizer(vectorizer, tfidf_matrix)