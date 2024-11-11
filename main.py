from website import create_app
from flask_compress import Compress
import os

app = create_app()
Compress(app)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
