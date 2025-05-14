from flask import Flask, request, render_template
import numpy as np
import cv2
import base64
from color_detector import get_color_palette, closest_color_name, classify_scene, palette_to_mood

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    original_img_b64 = None
    palette = None
    color_names = None
    mood = None
    scene_label = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            in_memory = file.read()
            npimg = np.frombuffer(in_memory, dtype=np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            # Extract colors
            palette = get_color_palette(image, k=5)
            color_names = [closest_color_name(c) for c in palette]
            mood = palette_to_mood(palette)
            scene_label = classify_scene(image)

            _, buffer = cv2.imencode('.jpg', image)
            original_img_b64 = base64.b64encode(buffer).decode('utf-8')

    return render_template('index.html',
                           original_img=original_img_b64,
                           palette=palette,
                           color_names=color_names,
                           mood=mood,
                           scene_label=scene_label)

if __name__ == '__main__':
    print('🔵 Starting Dominant-Color Detector on http://127.0.0.1:5000/')
    app.run(host='127.0.0.1', port=5000, debug=True)
