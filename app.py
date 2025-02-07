from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = tf.keras.models.load_model("model_rempah.h5")

label_map = {0: "Jahe", 1: "Kencur", 2: "Kunyit", 3: "Lengkuas"}

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Preprocess Gambar
        img = image.load_img(filepath, target_size=(64, 64))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediksi
        pred = model.predict(img)
        rempah_pred = label_map[np.argmax(pred)]

        return render_template("index.html", image=filepath, rempah=rempah_pred)

    return render_template("index.html", image=None, rempah=None)

if __name__ == "__main__":
    app.run(debug=True)

