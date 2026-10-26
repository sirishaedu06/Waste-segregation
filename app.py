
from flask import Flask, request, render_template_string
from PIL import Image, ImageStat, ImageFilter
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import io
import re

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

TIPS = {
    "Organic": "Place food scraps, fruit peels and vegetable waste in a wet-waste bin for composting.",
    "Paper": "Keep paper clean and dry. Recycle it separately from wet waste.",
    "Plastic": "Empty and clean plastic containers before placing them in the dry-recyclables bin.",
    "Metal": "Separate clean metal cans and containers for metal recycling.",
    "Glass": "Keep glass separate. Handle broken glass carefully and follow local collection rules.",
    "General Waste": "If the item cannot be recycled or composted, put it in the appropriate residual-waste bin."
}

# ML demonstration model: trained on illustrative feature samples.
# It is not a real-world, professionally trained waste dataset.
rng = np.random.default_rng(42)
labels = list(TIPS.keys())
X, y = [], []

for label in labels:
    for _ in range(150):
        if label == "Organic":
            rgb = [rng.normal(105, 35), rng.normal(125, 35), rng.normal(65, 30)]
            texture = rng.uniform(15, 65)
        elif label == "Paper":
            rgb = [rng.normal(190, 30), rng.normal(180, 30), rng.normal(155, 35)]
            texture = rng.uniform(3, 30)
        elif label == "Plastic":
            rgb = [rng.uniform(30, 230) for _ in range(3)]
            texture = rng.uniform(2, 35)
        elif label == "Metal":
            v = rng.normal(145, 35)
            rgb = [v + rng.normal(0, 8) for _ in range(3)]
            texture = rng.uniform(5, 40)
        elif label == "Glass":
            v = rng.normal(130, 45)
            rgb = [v + rng.normal(0, 12) for _ in range(3)]
            texture = rng.uniform(2, 25)
        else:
            rgb = [rng.uniform(35, 180) for _ in range(3)]
            texture = rng.uniform(10, 75)

        X.append([*rgb, texture])
        y.append(label)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)


def extract_features(image):
    image = image.convert("RGB").resize((128, 128))
    stats = ImageStat.Stat(image)
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_mean = ImageStat.Stat(edges).mean[0]
    return [*stats.mean, edge_mean]


PAGE = """
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Waste Segregation</title>
<style>
body {font-family:Arial,sans-serif;background:#eef7f0;margin:0;padding:18px}
main {max-width:600px;margin:20px auto;background:white;padding:22px;border-radius:16px}
h1 {color:#176b42}
button {background:#198754;color:white;border:0;padding:13px;
border-radius:8px;width:100%;font-size:16px;margin-top:12px}
input {margin:12px 0;max-width:100%}
.result {background:#e3f5e8;padding:14px;border-radius:10px;margin-top:18px}
img {max-width:100%;max-height:280px;object-fit:contain}
.small {color:#555;font-size:14px}
</style>
</head>
<body>
<main>
<h1>♻️ Smart Waste Segregation</h1>
<p>Upload a waste image to see a predicted category and disposal tips.</p>
<form method="post" enctype="multipart/form-data">
<input type="file" name="photo" accept="image/*" required>
<button type="submit">Analyze Waste</button>
</form>
{% if error %}<p style="color:red">{{ error }}</p>{% endif %}
{% if result %}
<div class="result">
<h2>Predicted category: {{ result }}</h2>
<p><b>Disposal advice:</b> {{ tip }}</p>
<p><b>DIP image features:</b> RGB color averages and edge intensity analyzed.</p>
<p class="small">Educational ML prototype. Verify the result before disposal.</p>
<img src="data:image/jpeg;base64,{{ image_data }}" alt="Uploaded waste">
</div>
{% endif %}
</main>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template_string(PAGE)

    photo = request.files.get("photo")
    if not photo or not photo.filename:
        return render_template_string(PAGE, error="Please select an image.")

    if not re.search(r"\.(jpg|jpeg|png|webp|bmp)$",
                     photo.filename, re.IGNORECASE):
        return render_template_string(
            PAGE, error="Please upload a JPG, PNG, WEBP or BMP image."
        )

    try:
        raw = photo.read()
        image = Image.open(io.BytesIO(raw))
        image.verify()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        features = extract_features(image)
        prediction = model.predict([features])[0]

        # Return the uploaded image for display.
        from base64 import b64encode
        preview = image.copy()
        preview.thumbnail((500, 500))
        output = io.BytesIO()
        preview.save(output, format="JPEG")
        image_data = b64encode(output.getvalue()).decode("utf-8")

        return render_template_string(
            PAGE, result=prediction, tip=TIPS[prediction],
            image_data=image_data
        )
    except Exception:
        return render_template_string(
            PAGE, error="Could not read that image. Try another JPG or PNG."
        )


if __name__ == "__main__":
    app.run()
