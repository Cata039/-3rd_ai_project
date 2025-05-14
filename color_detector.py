import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

HEX_TO_NAMES = {
    '#f0f8ff': 'aliceblue', '#faebd7': 'antiquewhite', '#00ffff': 'aqua', '#7fffd4': 'aquamarine',
    '#f0ffff': 'azure', '#f5f5dc': 'beige', '#ffe4c4': 'bisque', '#000000': 'black',
    '#ffebcd': 'blanchedalmond', '#0000ff': 'blue', '#8a2be2': 'blueviolet', '#a52a2a': 'brown',
    '#deb887': 'burlywood', '#5f9ea0': 'cadetblue', '#7fff00': 'chartreuse', '#d2691e': 'chocolate',
    '#ff7f50': 'coral', '#6495ed': 'cornflowerblue', '#fff8dc': 'cornsilk', '#dc143c': 'crimson',
    '#00ffff': 'cyan', '#00008b': 'darkblue', '#008b8b': 'darkcyan', '#b8860b': 'darkgoldenrod',
    '#a9a9a9': 'darkgray', '#006400': 'darkgreen', '#bdb76b': 'darkkhaki', '#8b008b': 'darkmagenta',
    '#556b2f': 'darkolivegreen', '#ff8c00': 'darkorange', '#9932cc': 'darkorchid', '#8b0000': 'darkred',
    '#e9967a': 'darksalmon', '#8fbc8f': 'darkseagreen', '#483d8b': 'darkslateblue', '#2f4f4f': 'darkslategray',
    '#00ced1': 'darkturquoise', '#9400d3': 'darkviolet', '#ff1493': 'deeppink', '#00bfff': 'deepskyblue',
    '#696969': 'dimgray', '#1e90ff': 'dodgerblue', '#b22222': 'firebrick', '#fffaf0': 'floralwhite',
    '#228b22': 'forestgreen', '#ff00ff': 'fuchsia', '#dcdcdc': 'gainsboro', '#f8f8ff': 'ghostwhite',
    '#ffd700': 'gold', '#daa520': 'goldenrod', '#808080': 'gray', '#008000': 'green',
    '#adff2f': 'greenyellow', '#f0fff0': 'honeydew', '#ff69b4': 'hotpink', '#cd5c5c': 'indianred',
    '#4b0082': 'indigo', '#fffff0': 'ivory', '#f0e68c': 'khaki', '#e6e6fa': 'lavender',
    '#fff0f5': 'lavenderblush', '#7cfc00': 'lawngreen', '#fffacd': 'lemonchiffon', '#add8e6': 'lightblue',
    '#f08080': 'lightcoral', '#e0ffff': 'lightcyan', '#fafad2': 'lightgoldenrodyellow', '#d3d3d3': 'lightgray',
    '#90ee90': 'lightgreen', '#ffb6c1': 'lightpink', '#ffa07a': 'lightsalmon', '#20b2aa': 'lightseagreen',
    '#87cefa': 'lightskyblue', '#778899': 'lightslategray', '#b0c4de': 'lightsteelblue', '#ffffe0': 'lightyellow',
    '#00ff00': 'lime', '#32cd32': 'limegreen', '#faf0e6': 'linen', '#ff00ff': 'magenta',
    '#800000': 'maroon', '#66cdaa': 'mediumaquamarine', '#0000cd': 'mediumblue', '#ba55d3': 'mediumorchid',
    '#9370db': 'mediumpurple', '#3cb371': 'mediumseagreen', '#7b68ee': 'mediumslateblue', '#00fa9a': 'mediumspringgreen',
    '#48d1cc': 'mediumturquoise', '#c71585': 'mediumvioletred', '#191970': 'midnightblue', '#f5fffa': 'mintcream',
    '#ffe4e1': 'mistyrose', '#ffe4b5': 'moccasin', '#ffdead': 'navajowhite', '#000080': 'navy',
    '#fdf5e6': 'oldlace', '#808000': 'olive', '#6b8e23': 'olivedrab', '#ffa500': 'orange',
    '#ff4500': 'orangered', '#da70d6': 'orchid', '#eee8aa': 'palegoldenrod', '#98fb98': 'palegreen',
    '#afeeee': 'paleturquoise', '#db7093': 'palevioletred', '#ffefd5': 'papayawhip', '#ffdab9': 'peachpuff',
    '#cd853f': 'peru', '#ffc0cb': 'pink', '#dda0dd': 'plum', '#b0e0e6': 'powderblue',
    '#800080': 'purple', '#663399': 'rebeccapurple', '#ff0000': 'red', '#bc8f8f': 'rosybrown',
    '#4169e1': 'royalblue', '#8b4513': 'saddlebrown', '#fa8072': 'salmon', '#f4a460': 'sandybrown',
    '#2e8b57': 'seagreen', '#fff5ee': 'seashell', '#a0522d': 'sienna', '#c0c0c0': 'silver',
    '#87ceeb': 'skyblue', '#6a5acd': 'slateblue', '#708090': 'slategray', '#fffafa': 'snow',
    '#00ff7f': 'springgreen', '#4682b4': 'steelblue', '#d2b48c': 'tan', '#008080': 'teal',
    '#d8bfd8': 'thistle', '#ff6347': 'tomato', '#40e0d0': 'turquoise', '#ee82ee': 'violet',
    '#f5deb3': 'wheat', '#ffffff': 'white', '#f5f5f5': 'whitesmoke', '#ffff00': 'yellow',
    '#9acd32': 'yellowgreen'}


clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_color_palette(image: np.ndarray, k: int = 10, resize_dim: int = 200) -> list[tuple[int, int, int]]:
    small = cv2.resize(image, (resize_dim, resize_dim), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    rgb_pixels = rgb.reshape(-1, 3)
    hsv_pixels = hsv.reshape(-1, 3)

    mask = (hsv_pixels[:, 1] > 50) & (hsv_pixels[:, 2] > 50)
    filtered_rgb = rgb_pixels[mask]

    if len(filtered_rgb) < k:
        filtered_rgb = rgb_pixels

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(filtered_rgb)
    centers = np.array(kmeans.cluster_centers_, dtype=np.uint8)
    labels = kmeans.labels_

    counts = np.bincount(labels)
    sorted_by_count = sorted(zip(counts, centers), reverse=True, key=lambda x: x[0])

    final_colors = []
    used_hues = []

    for count, rgb_color in sorted_by_count:
        hsv_color = cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2HSV)[0][0]
        hue = hsv_color[0]
        if all(abs(hue - h) > 15 for h in used_hues):
            final_colors.append(tuple(int(c) for c in rgb_color))
            used_hues.append(hue)
        if len(final_colors) == 5:
            break

    return final_colors

def closest_color_name(requested_color: tuple[int, int, int]) -> str:
    min_dist = float('inf')
    best_name = None
    for hex_code, name in HEX_TO_NAMES.items():
        r, g, b = webcolors.hex_to_rgb(hex_code)
        d = (r - requested_color[0])**2 + (g - requested_color[1])**2 + (b - requested_color[2])**2
        if d < min_dist:
            min_dist, best_name = d, name
    return best_name

def palette_to_mood(colors: list[tuple[int, int, int]]) -> str:
    color_names = [closest_color_name(c) for c in colors]
    joined = " ".join(color_names)

    red_variants = ["red", "firebrick", "crimson", "darkred", "indianred"]
    if any(r in joined for r in red_variants) and "black" in joined:
        return "Intense / Passionate"
    elif "blue" in joined and "gray" in joined:
        return "Calm / Melancholy"
    elif "green" in joined:
        return "Fresh / Natural"
    elif "yellow" in joined or "orange" in joined:
        return "Warm / Energetic"
    else:
        return "Neutral / Undefined"

def classify_scene(image: np.ndarray) -> str:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    texts = [
    "a tropical rainforest with a waterfall",
    "a mountain landscape",
    "a beach or ocean view",
    "a city skyline or architecture",
    "a plate of colorful food",
    "a close-up portrait of a person",
    "an indoor cluttered room",
    "an abstract or surreal painting"
]


    inputs = clip_processor(text=texts, images=pil_image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    best_idx = probs.argmax().item()
    return texts[best_idx]