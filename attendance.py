from flask import Flask, request, jsonify, render_template
import numpy as np
import face_recognition
import pickle
import os
from datetime import datetime
import requests
import mysql.connector
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# ---------- MySQL Connection ----------
def get_db_connection():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="",  # your MySQL password
        database="attendance_system",
        port=3306
    )

# ---------- Load encodings from DB ----------
def load_encodings_from_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT emp_id, name, face_encoding FROM employees")
    rows = cursor.fetchall()
    conn.close()

    encodings = []
    names = []
    for emp_id, name, face_encoding in rows:
        try:
            encoding = pickle.loads(face_encoding)
            encodings.append(encoding)
            names.append(f"{emp_id}_{name.replace(' ', '_')}")
        except Exception as e:
            print(f"[Error decoding face encoding for {emp_id}]: {e}")

    return encodings, names

known_encodings, known_names = load_encodings_from_db()

# ---------- Get Location ----------
def get_location():
    try:
        response = requests.get("http://ipinfo.io/json", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"{data.get('city')}, {data.get('country')}"
    except requests.exceptions.RequestException as e:
        print(f"[Location Error] {e}")
    return "Unknown Location"

# ---------- Parse Name & ID ----------
def parse_name_id(full_name):
    parts = full_name.split("_")
    emp_id = parts[0]
    name = " ".join(parts[1:])
    return name, emp_id

# ---------- Mark Attendance in DB ----------
def mark_attendance_db(full_name, mode):
    name, emp_id = parse_name_id(full_name)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    location = get_location()

    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if an entry exists for this employee today
    cursor.execute(
        "SELECT id, login_time, logout_time FROM attendance WHERE emp_id=%s AND date=%s",
        (emp_id, date_str)
    )
    record = cursor.fetchone()

    if record:
        attendance_id, login_time, logout_time = record
        if mode.lower() == "logout" and not logout_time:
            cursor.execute(
                "UPDATE attendance SET logout_time=%s WHERE id=%s",
                (time_str, attendance_id)
            )
            conn.commit()
            conn.close()
            return True, name, time_str
    else:
        if mode.lower() == "login":
            cursor.execute(
                "INSERT INTO attendance (emp_id, date, login_time, location) VALUES (%s, %s, %s, %s)",
                (emp_id, date_str, time_str, location)
            )
            conn.commit()
            conn.close()
            return True, name, time_str

    conn.close()
    return False, name, time_str

# ---------- Routes ----------
@app.route('/')
def index():
    return render_template("attendance.html")

@app.route('/recognize', methods=['POST'])
def recognize():
    if "image" not in request.files or "mode" not in request.form:
        return jsonify({"status": "error", "message": "Missing image or mode"}), 400

    img_file = request.files["image"]
    mode = request.form["mode"]

    # Load image from memory
    image = Image.open(img_file.stream).convert("RGB")
    image_np = np.array(image)

    # Face Recognition
    boxes = face_recognition.face_locations(image_np)
    encodings = face_recognition.face_encodings(image_np, boxes)

    for encoding in encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding)
        face_distances = face_recognition.face_distance(known_encodings, encoding)
        name = "Unknown"

        if len(face_distances) > 0:
            best_match = np.argmin(face_distances)
            if matches[best_match]:
                name = known_names[best_match]

        if name != "Unknown":
            success, real_name, time_str = mark_attendance_db(name, mode)
            if success:
                return jsonify({
                    "status": "success",
                    "message": f"{mode} marked for {real_name} at {time_str}",
                    "employee": real_name,
                    "mode": mode
                }), 200
            else:
                return jsonify({
                    "status": "fail",
                    "message": f"{mode} already marked for {real_name}"
                }), 409
        else:
            return jsonify({"status": "fail", "message": "Face not recognized"}), 401

    return jsonify({"status": "fail", "message": "No face found"}), 400

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "running", "message": "API is active"}), 200

if __name__ == "__main__":
    app.run(port=8000, debug=True)
