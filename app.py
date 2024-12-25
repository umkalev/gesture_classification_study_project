from flask import Flask, request, render_template
from transformers import pipeline
from PIL import Image
import io

app = Flask(__name__)

# Загрузка модели для классификации жестов
model_name = "dima806/hand_gestures_image_detection"
classifier = pipeline("image-classification", model=model_name)

# Список классов
classes = [
    "call", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm",
    "peace", "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2",
    "two_up", "two_up_inverted"
]

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "Файл не загружен"
        file = request.files["file"]
        if file.filename == "":
            return "Файл не выбран"
        if file:
            # Преобразование файла в изображение
            image = Image.open(io.BytesIO(file.read()))
            # Классификация изображения
            result = classifier(image)
            # Получение лучшего результата
            best_result = result[0]
            label = best_result["label"]
            score = best_result["score"]
            return render_template("result.html", label=label, score=score, classes=classes)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)