from flask import Flask, request, jsonify, render_template
import pickle
import face_recognition
import numpy as np
import mysql.connector
from datetime import datetime
from PIL import Image
import io

app = Flask(__name__)

# ---------- MySQL Connection ----------
def get_db_connection():
    return mysql.connector.connect(
        host="****",
        user="*****",
        password="",
        database="attendance_system",
        port=3306
    )

@app.route('/')
def home():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register_user():
    name = request.form.get("name")
    emp_id = request.form.get("emp_id")
    images = request.files.getlist("images")

    if not name or not emp_id or len(images) != 3:
        return jsonify({"status": "error", "message": "Missing name, emp_id, or not exactly 3 images"}), 400

    # Take first image encoding for DB storage
    first_encoding = None

    for i, img_file in enumerate(images, 1):
        # Read image directly from memory
        image_bytes = img_file.read()
        image = face_recognition.load_image_file(io.BytesIO(image_bytes))
        encodings = face_recognition.face_encodings(image)

        if encodings:
            if first_encoding is None:
                first_encoding = encodings[0]
        else:
            return jsonify({"status": "error", "message": f"No face detected in image {i}"}), 400

    # Store into DB
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO employees (emp_id, name, face_encoding, created_at) VALUES (%s, %s, %s, %s)",
            (
                emp_id,
                name,
                pickle.dumps(first_encoding),  # Serialize encoding
                datetime.now()
            )
        )

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        return jsonify({"status": "error", "message": f"Database error: {str(e)}"}), 500

    return jsonify({"status": "success", "message": f"Registered {name} successfully."}), 200

if __name__ == '__main__':
    app.run(debug=True, port=9000)

