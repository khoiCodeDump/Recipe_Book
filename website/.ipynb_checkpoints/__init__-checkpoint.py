from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager
import ast
import pickle 
import pandas as pd

db = SQLAlchemy()
DB_NAME = "database.db"

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)

    from .views import views
    from .auth import auth

    
    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')
    
    from .models import User 

    create_database(app)
    
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    return app


def create_database(app):
     if not path.exists('instance/' + DB_NAME):
        from .models import Recipe, Tag, Ingredient

        with app.app_context():
            tags = {}
            ingredients = {}
            db.create_all()
            raw_recipes = pd.read_csv('RAW_recipes.csv')
    
            for index, row in raw_recipes.iterrows():
                ingredients_list = ast.literal_eval(row["ingredients"])
                steps_list = ast.literal_eval(row["steps"])
                tags_list = ast.literal_eval(row["tags"])
    
                steps_list = '|'.join(steps_list)
                
                recipe = Recipe(name = row["name"], recipe_id = int(row["id"]), cook_time = int(row["minutes"]),steps = steps_list, desc = row["description"])
                
                db_tag_list = []
                db_ingredient_list = []
                
                for ingredient in ingredients_list:
                    if ingredient not in ingredients:
                        db_ingredient = Ingredient(name=ingredient)
                        ingredients[ingredient] = db_ingredient
                        db_ingredient_list = db_ingredient_list + [db_ingredient]
                    else:
                        db_ingredient = ingredients[ingredient]

                    recipe.ingredients.append(db_ingredient)
                    
                for tag in tags_list:
                    if tag not in ingredients:
                        if tag not in tags:
                            db_tag = Tag(name=tag)
                            tags[tag] = db_tag
                            db_tag_list = db_tag_list + [db_tag]
                        else:
                            db_tag = tags[tag]
                            
                        recipe.tags.append(db_tag)

                db.session.add_all(db_tag_list)
                db.session.add_all(db_ingredient_list)
                db.session.add(recipe)
                db.session.commit()
               
            print('Created Database!')
            with open('tags_dict.pkl', 'wb') as f:
                pickle.dump(tags, f)
            with open('ingredients_dict.pkl', 'wb') as f:
                pickle.dump(ingredients, f)