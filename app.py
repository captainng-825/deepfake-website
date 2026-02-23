from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import shutil
import cv2
from transformers import pipeline
from PIL import Image
from insightface.app import FaceAnalysis

# =========================
# FLASK SETUP
# =========================
app = Flask(__name__)

# ✅ Railway-safe secret key
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "devsecret")

# ✅ Railway-compatible database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    "DATABASE_URL",
    "sqlite:///users.db"
)

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# DATABASE MODEL
# =========================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# =========================
# LOAD AI MODELS (ONCE)
# =========================
print("🚀 Loading Deepfake Model...")

classifier = pipeline(
    "image-classification",
    model="dima806/deepfake_vs_real_image_detection",
    device=-1  # CPU mode
)

print("✅ Deepfake Model Loaded")

face_app = FaceAnalysis()

# ✅ Force CPU mode (Railway has no GPU)
face_app.prepare(ctx_id=-1, det_size=(640, 640))

print("✅ Face Detector Loaded")

# =========================
# HELPER FUNCTION
# =========================
def extract_scores(result):
    fake_score = 0
    real_score = 0

    for r in result:
        label = r["label"].lower()
        if "fake" in label:
            fake_score = r["score"]
        elif "real" in label:
            real_score = r["score"]

    return fake_score, real_score

# =========================
# ROUTES
# =========================
@app.route("/")
@login_required
def home():
    return render_template("index.html")

# =========================
# DETECT ROUTE
# =========================
@app.route("/detect", methods=["POST"])
@login_required
def detect():

    if 'file' not in request.files:
        return redirect(url_for("home"))

    file = request.files['file']

    if file.filename == "":
        return redirect(url_for("home"))

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    image = cv2.imread(file_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = face_app.get(rgb_image)

    if len(faces) == 0:
        return render_template("index.html",
            image_path=file_path,
            scores=[],
            face_count=0,
            verdict="❌ NO FACE DETECTED",
            avg_score=0
        )

    scores = []

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        face_crop = rgb_image[y1:y2, x1:x2]

        pil_face = Image.fromarray(face_crop)
        result = classifier(pil_face)

        fake_score, real_score = extract_scores(result)

        confidence_gap = fake_score - real_score

        if confidence_gap < 0.20:
            adjusted_fake = fake_score * 0.6
        else:
            adjusted_fake = fake_score

        scores.append(adjusted_fake)

        percent = round(adjusted_fake * 100, 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            image,
            f"{percent}%",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    annotated_path = os.path.join(UPLOAD_FOLDER, "annotated_" + file.filename)
    cv2.imwrite(annotated_path, image)

    scores_percent = [round(s * 100, 2) for s in scores]
    face_count = len(scores_percent)
    avg_score = sum(scores_percent) / face_count

    if avg_score >= 75:
        verdict = "⚠ FAKE"
    elif avg_score >= 50:
        verdict = "⚠ UNCERTAIN"
    else:
        verdict = "✅ REAL"

    return render_template("index.html",
        image_path=annotated_path,
        scores=scores_percent,
        face_count=face_count,
        verdict=verdict,
        avg_score=round(avg_score, 2)
    )

# =========================
# SIGNUP
# =========================
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already exists!")
            return redirect(url_for("signup"))

        hashed_password = generate_password_hash(password)
        new_user = User(email=email, password=hashed_password)

        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for("login"))

    return render_template("signup.html")

# =========================
# LOGIN
# =========================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("home"))
        else:
            flash("Invalid credentials!")

    return render_template("login.html")

# =========================
# LOGOUT
# =========================
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# =========================
# RUN (Railway Compatible)
# =========================
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)