from flask import abort, send_from_directory
from flask_login import current_user
import os



def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov', 'avi'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def serve_media(filename, media_type):
    media_dir = os.path.join(os.path.join('../data', str(current_user.id)), media_type)
    try:
        return send_from_directory(media_dir, filename)
    except FileNotFoundError as e:
        abort(404)



