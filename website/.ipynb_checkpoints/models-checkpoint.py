from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func

table = db.Table('recipe_tag',
                    db.Column('recipe_id', db.Integer, db.ForeignKey('recipe.id')),
                    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id')),
                    # db.Column('ingredient_id', db.Integer, db.ForeignKey('ingredient.id'))
                )
ingredient_table = db.Table('recipe_ingredient',
                    db.Column('recipe_id', db.Integer, db.ForeignKey('recipe.id')),
                    db.Column('ingredient_id', db.Integer, db.ForeignKey('ingredient.id'))
                )

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))    
    recipe_id = db.relationship('Recipe', backref='user')

class Recipe(db.Model):
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(1000))
    recipe_id = db.Column(db.Integer)
    cook_time = db.Column(db.Integer)
    ingredients = db.Column(db.String)
    steps = db.Column(db.String)
    tags = db.relationship('Tag', secondary=table, backref='recipes')
    ingredients = db.relationship('Ingredient', secondary=ingredient_table, backref='recipes')
    desc = db.Column(db.String)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    
class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(1000))

class Ingredient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(1000))
    