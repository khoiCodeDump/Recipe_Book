from flask import Blueprint, render_template, request, flash, jsonify, redirect, url_for
from flask_login import login_required, current_user
from .models import User, Recipe, Tag, Ingredient, table, ingredient_table
from . import db
import json
import pickle
import math
import os

views = Blueprint('views', __name__)

with open('tags_dict.pkl', 'rb') as f:
    tags = pickle.load(f)

with open('ingredients_dict.pkl', 'rb') as f:
    ingredients = pickle.load(f)


@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    global tags
    page = request.args.get('page', 1, type=int)
    pagination = Recipe.query.paginate(page=page, per_page=20)
    return render_template("home.html", allrecipes = pagination, tags=list(tags.keys()), ingredients=list(ingredients.keys()) )
    
@views.route('/profile', methods=['GET'])
def profile():
    page = request.args.get('page', 1, type=int)
    pagination = Recipe.query.filter(Recipe.user_id==current_user.id).paginate(page=page, per_page=20)
    return render_template("profile.html", user=current_user, allrecipes=pagination) 

# @views.route('/edit_recipe', methods=['GET, POST'])
# def edit_recipe():
    
#     return render_template("post_recipe_form.html", user=current_user, tags=list(tags.keys()), ingredients=list(ingredients.keys()) )
    
@views.route('/post_recipe', methods=['GET', 'POST'])
def post_recipe():
    global tags, ingredients
    if request.method == 'POST': 
        name = request.form.get("Title")
        cook_time = request.form.get("Cook_time")
        desc = request.form.get("Description")
        steps = request.form.get("Instructions")
        tags_db = request.form.get("Tags")
        ingredients_db = request.form.get("Ingredients")
        tags_db = tags_db.split(",")
        ingredients_db = ingredients_db.split(",")

        # https://en.wikipedia.org/wiki/Newline
        temp = steps.split("\r\n")
        for step in temp:
            step = step.strip()
        steps = "|".join(temp)
        
        
        recipe =  Recipe(name = name, cook_time = cook_time,steps = steps, desc = desc)
        new_tags = []
        new_ings = []
        for tag in tags_db:
            tag = tag.strip()
            if tag in tags:
                queried_tag = Tag.query.filter(Tag.name==tag).first()
            else:
                queried_tag = Tag(name=tag)
                new_tags = new_tags + [queried_tag]
            recipe.tags.append(queried_tag)
            
        for ingredient in ingredients_db:
            ingredient = ingredient.strip()
            if ingredient in ingredients:
                queried_ing = Ingredient.query.filter(Ingredient.name==ingredient).first()
            else:
                queried_ing = Ingredient(name=ingredient)
                new_ings = new_ings + [queried_ing]
            recipe.ingredients.append(queried_ing)
            
        current_user.recipe_id.append(recipe)
        
        db.session.add_all(new_tags)
        db.session.add_all(new_ings)
        db.session.add(recipe)
        db.session.commit()
        return redirect( url_for('views.profile', code=302))
 
    return render_template("post_recipe_form.html", user=current_user, tags=list(tags.keys()), ingredients=list(ingredients.keys()))

@views.route('/recipes/<meal_id>', methods=['GET'])
def get_recipe(meal_id):
    cur_recipe =  Recipe.query.get(meal_id)

    temp = cur_recipe.steps
    typeList = "ol"
    if temp[0] >= "0" and temp[0] <= "9":
        typeList = "ul"
    return render_template("recipe_base.html", user=current_user, recipe_info = cur_recipe, typeList =typeList)

@views.route('/search', methods=['GET'])
def search():
    global tags, ingredients

    search_field = request.args.get('search-field')
        
    if search_field == '':
        return redirect(url_for('home'))

    # search_field = re.findall(r'\w+', search_field)
    search_field = search_field.split(",")
    
    recipe_results = []
    index = 0
    
    result = None
    recipes = set()
    clone_recipe_ids = set()
    
    for field in search_field:
        field = field.strip()
        if field in ingredients:
            print("In ingredients")
            ingredient_results = Ingredient.query.filter(Ingredient.name==field).first()
            
            if len(recipes) == 0:
                for recipe in ingredient_results.recipes:
                     recipes.add(recipe.recipe_id)
            else:
                for recipe in ingredient_results.recipes:
                    if recipe.recipe_id in recipes:
                        clone_recipe_ids.add(recipe.recipe_id)

                recipes = clone_recipe_ids
                clone_recipe_ids = set()

            if result ==None:
                result = Recipe.query.filter(Recipe.recipe_id.in_(list(recipes)))
            else:
                result = result.filter(Recipe.recipe_id.in_(list(recipes)))
                
        elif field in tags:
            tag_results = Tag.query.filter(Tag.name==field).first()
            
            if len(recipes) == 0:
                for recipe in tag_results.recipes:
                     recipes.add(recipe.recipe_id)
            else:
                for recipe in tag_results.recipes:
                    if recipe.recipe_id in recipes:
                        clone_recipe_ids.add(recipe.recipe_id)

                recipes = clone_recipe_ids
                clone_recipe_ids = set()

            if result ==None:
                result = Recipe.query.filter(Recipe.recipe_id.in_(list(recipes)))
            else:
                result = result.filter(Recipe.recipe_id.in_(list(recipes)))
                    
        else:
            search_results = Recipe.query.filter(Recipe.name.contains(field))
            
            if len(recipes) == 0:
                for recipe in search_results:
                     recipes.add(recipe.recipe_id)
            else:
                #the clone is the union of recipe_ids in both separate sets
                for recipe in search_results:
                    if recipe.recipe_id in recipes:
                        clone_recipe_ids.add(recipe.recipe_id)
                
                recipes = clone_recipe_ids
                clone_recipe_ids = set()
            
            if result ==None:
                result = Recipe.query.filter(Recipe.recipe_id.in_(recipes))
            else:
                result = result.filter(Recipe.recipe_id.in_(recipes))
                    

    search_recipes = []
    for result_recipe in result:
        search_recipes.append(result_recipe.recipe_id)

    if not os.path.exists('data/' + str(current_user.id)):
        os.mkdir('data/' + str(current_user.id))
    else:
        shutil.rmtree('data/' + str(current_user.id))
        os.mkdir('data/' + str(current_user.id))

    with open('data/' + str(current_user.id) + '/result.pkl', 'wb') as f:
        pickle.dump(search_recipes, f)
                    
    return redirect( url_for('views.search_pagination', search_field=",".join(search_field)), code=302)

@views.route('/page', methods=['GET'])
def search_pagination():
    global tags
    page = request.args.get('page', 1, type=int)
    search_field = request.args.get('search_field')
    
    with open('data/' + str(current_user.id) + '/result.pkl', 'rb') as f:
        recipe_list = pickle.load(f)
        
    pagination = Recipe.query.filter(Recipe.recipe_id.in_(recipe_list)).paginate(page=page, per_page=20)

    return render_template("search_view.html", user=current_user, search_field=search_field, pagination=pagination, tags=list(tags.keys()), ingredients=list(ingredients.keys()) )


@views.route('/delete_recipe', methods=['POST'])
def delete_recipe():

    recipe = json.loads(request.data)
    recipe_id = recipe['recipe']
    recipe = Recipe.query.get(recipe_id)
        
    if recipe:
        if recipe.user_id == current_user.id:
            db.session.delete(recipe)
            db.session.commit()
        
    return jsonify({})