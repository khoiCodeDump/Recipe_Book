from flask import Blueprint, render_template, request, jsonify, redirect, url_for, make_response, abort, current_app, send_file, session
from flask_login import login_required, current_user
from .models import Recipe, Tag, Ingredient, Image, Video, add_recipe_to_faiss, remove_recipe_from_faiss, combined_search_recipes
from . import db
from werkzeug.utils import secure_filename
from .utils import allowed_file
import json
import os
import time
import bleach
from website import cache, model_name
from sentence_transformers import SentenceTransformer
import asyncio
import re
import uuid


views = Blueprint('views', __name__)
quantity = 45
model = SentenceTransformer(model_name)

RETRY_DELAY = 1  # seconds

@views.route('/', methods=['GET', 'POST'])
def home():
    data = {"route": 0}
    return render_template("home.html", data=data )

@views.route('/load', methods=['GET'])
def load():
    time.sleep(0.2)
    count = request.args.get('count', 0, type=int)

    try:
        ids = [id + 1 for id in range(count, count+quantity)]
        res = Recipe.query.filter(Recipe.id.in_(ids)).all()
        data = {}
        for stuff in res:
            first_image = stuff.images.first()
            image_url = url_for('views.serve_image', filename=first_image.filename) if first_image else url_for('static', filename='images/food_image_empty.png')

            data[stuff.id] = {
                'name': stuff.name,
                'image_url': image_url,
                'id' : stuff.id
            }
        res = make_response(data)
    except Exception as e:
        print(f"Error in load route: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        res = make_response(jsonify({}), 200)

    return res

@views.route('/images/<path:filename>', methods=['GET'])
async def serve_image(filename):
    try:
        file_path = os.path.join(current_app.root_path, 'images', filename)

        current_app.logger.info(f"Attempting to serve image: {file_path}")

        if not os.path.exists(file_path):
            current_app.logger.error(f"File not found: {file_path}")
            abort(404)

        file_size = os.path.getsize(file_path)
        range_header = request.headers.get('Range', None)

        if range_header:
            byte1, byte2 = 0, None
            match = re.search(r'(\d+)-(\d*)', range_header)
            groups = match.groups()

            if groups[0]:
                byte1 = int(groups[0])
            if groups[1]:
                byte2 = int(groups[1])

            if byte2 is None:
                byte2 = file_size - 1
            length = byte2 - byte1 + 1

            loop = asyncio.get_event_loop()
            def read_file_chunk():
                with open(file_path, 'rb') as f:
                    f.seek(byte1)
                    return f.read(length)

            chunk = await loop.run_in_executor(None, read_file_chunk)

            resp = current_app.response_class(
                chunk,
                206,
                mimetype=f'image/{os.path.splitext(filename)[1][1:]}',
                direct_passthrough=True,
            )
            resp.headers.set('Content-Range', f'bytes {byte1}-{byte2}/{file_size}')
            resp.headers.set('Accept-Ranges', 'bytes')
            resp.headers.set('Content-Length', str(length))
        else:
            resp = send_file(file_path)

        return resp

    except Exception as e:
        current_app.logger.error(f"Error serving image {filename}: {str(e)}")
        abort(500)

@views.route('/videos/<path:filename>', methods=['GET'])
async def serve_video(filename):
    try:
        file_path = os.path.join(current_app.root_path, 'videos', filename)

        current_app.logger.info(f"Attempting to serve video: {file_path}")

        if not os.path.exists(file_path):
            current_app.logger.error(f"File not found: {file_path}")
            abort(404)

        file_size = os.path.getsize(file_path)
        range_header = request.headers.get('Range', None)

        if range_header:
            byte1, byte2 = 0, None
            match = re.search(r'(\d+)-(\d*)', range_header)
            groups = match.groups()

            if groups[0]:
                byte1 = int(groups[0])
            if groups[1]:
                byte2 = int(groups[1])

            if byte2 is None:
                byte2 = file_size - 1
            length = byte2 - byte1 + 1

            loop = asyncio.get_event_loop()
            def read_file_chunk():
                with open(file_path, 'rb') as f:
                    f.seek(byte1)
                    return f.read(length)

            chunk = await loop.run_in_executor(None, read_file_chunk)

            resp = current_app.response_class(
                chunk,
                206,
                mimetype=f'video/{os.path.splitext(filename)[1][1:]}',
                direct_passthrough=True,
            )
            resp.headers.set('Content-Range', f'bytes {byte1}-{byte2}/{file_size}')
            resp.headers.set('Accept-Ranges', 'bytes')
            resp.headers.set('Content-Length', str(length))
        else:
            def read_file():
                with open(file_path, 'rb') as f:
                    return f.read()

            loop = asyncio.get_event_loop()
            file_content = await loop.run_in_executor(None, read_file)

            resp = current_app.response_class(
                file_content,
                200,
                mimetype=f'video/{os.path.splitext(filename)[1][1:]}',
                direct_passthrough=True,
            )
            resp.headers.set('Content-Length', str(file_size))

        return resp

    except Exception as e:
        current_app.logger.error(f"Error serving video {filename}: {str(e)}")
        abort(500)

@views.route('/recipes/<recipe_id>', methods=['GET'])
def get_recipe(recipe_id):
    cur_recipe = Recipe.query.get(recipe_id)
    temp = cur_recipe.steps
    typeList = "ol"
    if temp:
        if temp[0] >= "0" and temp[0] <= "9":
            typeList = "ul"

    # Get all valid image URLs for this recipe
    image_urls = []
    images_to_remove = []
    for image in cur_recipe.images:
        if os.path.exists(image.url):
            image_urls.append(url_for('views.serve_image', filename=image.filename))
        else:
            images_to_remove.append(image)

    # Get all valid video URLs for this recipe
    video_urls = []
    videos_to_remove = []
    for video in cur_recipe.videos:
        if os.path.exists(video.url):
            video_urls.append(url_for('views.serve_video', filename=video.filename))
        else:
            videos_to_remove.append(video)

    # Update the recipe references by removing non-existent files
    if images_to_remove or videos_to_remove:
        for image in images_to_remove:
            cur_recipe.images.remove(image)
            db.session.delete(image)
        for video in videos_to_remove:
            cur_recipe.videos.remove(video)
            db.session.delete(video)
        db.session.commit()

    return render_template("recipe_base.html",
                           user=current_user,
                           recipe_info=cur_recipe,
                           typeList=typeList,
                           image_urls=image_urls,
                           video_urls=video_urls)

@views.route('/search', methods=['POST'])
def search():
    user_query = bleach.clean(request.form.get("search-field"))

    if not user_query:
        return redirect(request.referrer or url_for('views.home'))

    all_recipes_ids_len = cache.get('all_recipes_ids_len')

    if current_user.is_authenticated:
        user_id = current_user.id
    else:
        if 'anonymous_id' not in session:
            session['anonymous_id'] = str(uuid.uuid4())
        user_id = session['anonymous_id']

    cache.set(f"user:{user_id}:search_result", None)

    results_ids = combined_search_recipes(user_query=user_query, k_elements=all_recipes_ids_len)

    
    cache.set(f"user:{user_id}:search_result", results_ids)
    # Return the search results to the user
    data = {"route": 2}

    return render_template("search_view.html", data=data, search_field= user_query)


@views.route('/load_search', methods=['GET'])
def load_search():

    time.sleep(0.2)
    count = request.args.get('count', 0, type=int)
    try:
        if current_user.is_authenticated:
            user_id = current_user.id
        else:
            user_id = session.get('anonymous_id')

        recipe_list = cache.get(f"user:{user_id}:search_result")
        
        res = Recipe.query.filter(Recipe.id.in_(recipe_list[count: count + quantity])).order_by(
            db.case({id: index for index, id in enumerate(recipe_list[count: count + quantity])}, value=Recipe.id)
        )

        data = {}

        for i, recipe in enumerate(res):
            first_image = recipe.images.first()
            image_url = url_for('views.serve_image', filename=first_image.filename) if first_image else url_for('static', filename='images/food_image_empty.png')
            data[i] = {
                'id' : recipe.id,
                'name': recipe.name,
                'image_url': image_url
            }
        res = make_response(data)
    except Exception as e:
        print(f"Error loading posts: {str(e)}")
        res = make_response(jsonify({}), 200)

    return res

@views.route('/profile', methods=['GET'])
@login_required
def profile():
    data = {"route": 1}
    if not cache.get(f"user:{current_user.id}:profile"):
        ids = [recipe.id for recipe in Recipe.query.filter(Recipe.user_id==current_user.id).all()]
        cache.set(f"user:{current_user.id}:profile", ids)

    return render_template("profile.html", data=data)

@views.route('/load_profile', methods=['GET'])
@login_required
def load_profile():
    time.sleep(0.2)
    count = request.args.get('count', 0, type=int)

    try:
        recipe_ids = cache.get(f"user:{current_user.id}:profile")

        res = Recipe.query.filter(Recipe.id.in_(recipe_ids[count: count + quantity]))
        res = res[count: count + quantity]
        data = {}
        for stuff in res:
            first_image = stuff.images.first()
            image_url = url_for('views.serve_image', filename=first_image.filename) if first_image else url_for('static', filename='images/food_image_empty.png')
            data[stuff.id] = {
                'name': stuff.name,
                'image_url': image_url,
                'id' : stuff.id
            }
        res = make_response(data)
    except:
        print("No more posts")
        res = make_response(jsonify({}), 200)

    return res

@views.route('/post_recipe', methods=['GET', 'POST'])
@login_required
def post_recipe():

    if request.method == 'POST':
        try:
                
            recipe_id = request.form.get('recipe_id')
            if recipe_id:
                # Updating existing recipe
                recipe = Recipe.query.filter_by(id=recipe_id, user_id=current_user.id).first()
                if not recipe:
                    abort(403)  # Forbidden if the recipe doesn't exist or doesn't belong to the user
            else:
                # Creating new recipe
                recipe = Recipe(user_id=current_user.id)
                db.session.add(recipe)

            # Update recipe fields
            recipe.name = bleach.clean(request.form.get("Title"))
            recipe.cook_time = bleach.clean(request.form.get("Cook_time"))
            recipe.desc = bleach.clean(request.form.get("Description"))

            # Handle steps
            steps = bleach.clean(request.form.get("Instructions"))
            recipe.steps = steps  # This will now be a |-separated string of steps


            # Handle existing images
            existing_images = [value for key, value in request.form.items() if key.startswith('existing_images_')]
            for image in recipe.images:
                if image.filename in existing_images:
                    existing_images.remove(image.filename)
                else:
                    os.remove(image.url)
                    db.session.delete(image)

            # Ensure images directory exists
            image_dir = os.path.join(current_app.root_path, 'images')
            # Handle new image uploads
            new_images = []
            for key, value in request.files.items():
                if key.startswith('new_images_'):
                    new_images.extend(request.files.getlist(key))
            for image in new_images:
                if image and allowed_file(image.filename):
                    filename = secure_filename(image.filename)
                    filename = f"{uuid.uuid4().hex}{os.path.splitext(filename)[1]}"
                    image_path = os.path.join(image_dir, filename)
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    image.save(image_path)
                    new_image = Image(filename=filename, url=image_path, recipe=recipe)
                    db.session.add(new_image)


            # Handle existing video
            existing_video = request.form.get('existing_video')
            if not existing_video:
                old_video = Video.query.filter_by(recipe_id=recipe.id).first()
                if old_video and os.path.exists(old_video.url):
                    os.remove(old_video.url)
                    db.session.delete(old_video)

            # Ensure videos directory exists
            video_dir = os.path.join(current_app.root_path, 'videos')
            # Handle new video upload
            new_video = request.files.get('new_video')
            if new_video and allowed_file(new_video.filename):
                filename = secure_filename(new_video.filename)
                filename = f"{uuid.uuid4().hex}{os.path.splitext(filename)[1]}"
                video_path = os.path.join(video_dir, filename)
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                new_video.save(video_path)
                new_video = Video(filename=filename, url=video_path, recipe=recipe)
                db.session.add(new_video)

            # Update tags and ingredients
            recipe.tags = []
            recipe.ingredients = []

            # Remove any empty strings
            tags_input = {tag.strip() for tag in set(request.form.get('TagsInput', '').split(','))}
            tags_component = set(request.form.get('Tags', '').split(','))
            tags = tags_input.union(tags_component)

            ingredients_input = {ingredient.strip() for ingredient in set(request.form.get('IngredientsInput', '').split(','))}
            ingredients_component = set(request.form.get('Ingredients', '').split(','))
            ingredients = ingredients_input.union(ingredients_component)
            # Fetch existing tags and ingredients in one query
            existing_tags = {tag.name: tag for tag in Tag.query.filter(Tag.name.in_(tags)).all()}
            existing_ingredients = {ingredient.name: ingredient for ingredient in Ingredient.query.filter(Ingredient.name.in_(ingredients)).all()}

            # Process tags
            for tag_name in tags:
                if not tag_name:
                    continue
                tag = existing_tags.get(tag_name)
                if not tag:
                    tag = Tag(name=tag_name)
                    db.session.add(tag)
                recipe.tags.append(tag)

            # Process ingredients
            for ingredient_name in ingredients:
                if not ingredient_name:
                    continue
                ingredient = existing_ingredients.get(ingredient_name)
                if not ingredient:
                    ingredient = Ingredient(name=ingredient_name)
                    db.session.add(ingredient)
                recipe.ingredients.append(ingredient)


            ingredients_text = ', '.join([ingredient.name for ingredient in recipe.ingredients])
            tags_text = ', '.join([tag.name for tag in recipe.tags])
            numbered_steps = " ".join([f"{i+1}. {step}" for i, step in enumerate(recipe.steps.split("|"))])

            text_data = (
                f"The recipe name is {recipe.name}"
                f"The recipe takes {recipe.cook_time} minutes to cook"
                f"To cook the recipe, the following ingredients are required, separated by commas: {ingredients_text}."
                f"The recipe has the following associated tags, separated by commas: {tags_text}."
                f"The description of the recipe is: {recipe.desc}. "
                f"Here are the instructions to cook the recipe: {numbered_steps}."
            )

            # Generate embedding for the recipe
            embedding = model.encode(text_data)
            recipe.embedding = embedding.tolist()

            #Commit changes to database
            db.session.commit()

            add_recipe_to_faiss(recipe=recipe)

            user_search_cache_key = f"user:{current_user.id}:search_result"
            user_profile_cache_key = f"user:{current_user.id}:profile"
            # Use delete_many for efficient multiple key deletion
            cache.delete(user_search_cache_key)
            cache.delete(user_profile_cache_key)
            cache.set('all_recipes_ids_len', Recipe.query.count())

            return redirect(url_for('views.get_recipe', recipe_id=recipe.id))
        except Exception as e:
            # Log the error
            current_app.logger.error(f"Error in post_recipe: {str(e)}")
            
            # Return error response
            return jsonify({
                'error': 'Failed to save recipe',
                'message': str(e)
            }), 500

    tags = Tag.query.all()
    ingredients = Ingredient.query.all()
    # GET request. Adding new recipe
    recipe_id = request.args.get('recipe_id')
    if not recipe_id:
        return render_template("post_recipe_form.html", user=current_user, recipe=[], tags=tags, ingredients=ingredients)

    # Updating existing recipe
    recipe = Recipe.query.filter_by(id=recipe_id, user_id=current_user.id).first()
    if not recipe:
        abort(403)  # Forbidden if the recipe doesn't exist or doesn't belong to the user
    existing_images = Image.query.filter_by(recipe_id=recipe.id).all()
    existing_video = Video.query.filter_by(recipe_id=recipe.id).first()

    return render_template("post_recipe_form.html", user=current_user, recipe=recipe, tags=tags, ingredients=ingredients, existing_images=existing_images, existing_video=existing_video)

@views.route('/delete_recipe', methods=['POST'])
@login_required
def delete_recipe():

    recipe = json.loads(request.data)
    recipe_id = recipe['recipe']
    recipe = Recipe.query.get(recipe_id)

    if recipe:
        if recipe.user_id == current_user.id:
            remove_recipe_from_faiss(recipe=recipe)
            for image in recipe.images:
                os.remove(image.url)
                db.session.delete(image)
            for video in recipe.videos:
                os.remove(video.url)
                db.session.delete(video)
            for tag in recipe.tags:
                if len(tag.recipes) == 1:  # Only this recipe is associated
                    db.session.delete(tag)
            for ingredient in recipe.ingredients:
                if len(ingredient.recipes) == 1:  # Only this recipe is associated
                    db.session.delete(ingredient)
            db.session.delete(recipe)
            db.session.commit()

    user_search_cache_key = f"user:{current_user.id}:search_result"
    user_profile_cache_key = f"user:{current_user.id}:profile"
    cache.delete(user_search_cache_key)
    cache.delete(user_profile_cache_key)
    cache.set('all_recipes_ids_len', Recipe.query.count())

    # remove_recipe_from_faiss(recipe_id=recipe_id)
    return jsonify({})


@views.route('/health')
def health_check():
    try:
        # Verify database connection
        from sqlalchemy import text
        db.session.execute(text('SELECT 1'))  # Simple query to test connection
        db_status = "connected"
    except Exception as e:
        db_status = "disconnected"
        return jsonify({
            "status": "unhealthy",
            "database": db_status,
            "error": str(e)
        }), 500

    return jsonify({
        "status": "healthy",
        "database": db_status
    }), 200