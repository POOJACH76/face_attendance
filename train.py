# app.py
from flask import Flask, request, jsonify, render_template
import os
import face_recognition
from PIL import Image
import numpy as np
import mysql.connector
from datetime import datetime, date
import requests
import pickle

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------- DB ----------
def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Pooja@123",
        database="pooja"
    )

# ---------- Utilities ----------
def get_location():
    try:
        r = requests.get("http://ipinfo.io/json", timeout=4)
        if r.ok:
            d = r.json()
            return f"{d.get('city')}, {d.get('country')}"
    except Exception:
        pass
    return "Unknown"

def average_encodings(enc_list):
    """Return the mean encoding (numpy array) of a list of encodings."""
    if not enc_list:
        return None
    arr = np.stack(enc_list, axis=0)
    return np.mean(arr, axis=0)

# ---------- Pages ----------
@app.route('/')
def index():
    return render_template('attendance.html')

@app.route('/register_page')
def register_page():
    return render_template('register.html')

# ---------- Registration ----------
@app.route('/register', methods=['POST'])
def register():
    name = request.form.get("name")
    emp_id = request.form.get("emp_id")
    images = request.files.getlist("images")

    if not name or not emp_id or len(images) != 3:
        return jsonify({"status": "error", "message": "Provide name, emp_id, and exactly 3 images"}), 400

    encodings_collected = []
    for img_file in images:
        img = Image.open(img_file).convert("RGB")
        img_np = np.array(img)
        encs = face_recognition.face_encodings(img_np)
        if not encs:
            return jsonify({"status": "error", "message": "Face not detected in one of the images"}), 400
        encodings_collected.append(encs[0])

    # average encodings to one encoding per employee
    mean_encoding = average_encodings(encodings_collected)
    encoding_blob = pickle.dumps(mean_encoding)

    conn = get_db()
    cursor = conn.cursor()
    # Upsert: if employee exists, update encoding & name; else insert
    try:
        cursor.execute("SELECT id FROM employees WHERE emp_id=%s", (emp_id,))
        row = cursor.fetchone()
        if row:
            cursor.execute("UPDATE employees SET name=%s, face_encoding=%s WHERE emp_id=%s",
                           (name, encoding_blob, emp_id))
        else:
            cursor.execute("INSERT INTO employees (emp_id, name, face_encoding) VALUES (%s, %s, %s)",
                           (emp_id, name, encoding_blob))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

    return jsonify({"status": "success", "message": f"{name} registered/updated successfully."}), 200

# ---------- Recognition / Attendance ----------
@app.route('/recognize', methods=['POST'])
def recognize():
    if "image" not in request.files or "mode" not in request.form:
        return jsonify({"status": "error", "message": "Missing image or mode"}), 400

    mode = request.form["mode"]
    img = Image.open(request.files["image"]).convert("RGB")
    img_np = np.array(img)
    encs = face_recognition.face_encodings(img_np)

    if not encs:
        return jsonify({"status": "fail", "message": "No face found"}), 400

    probe = encs[0]

    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT emp_id, name, face_encoding FROM employees")
    rows = cursor.fetchall()

    best_emp = None
    best_dist = None

    # Compare with all DB encodings (you can optimize/caching later)
    for r in rows:
        db_enc = pickle.loads(r['face_encoding'])
        # face_distance returns a scalar when comparing single encoding
        dist = face_recognition.face_distance([db_enc], probe)[0]
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_emp = (r['emp_id'], r['name'], db_enc)

    # threshold: choose 0.45-0.6 depending on your model; adjust after testing
    THRESHOLD = 0.50
    if best_emp and best_dist is not None and best_dist <= THRESHOLD:
        emp_id, name, _ = best_emp
        now = datetime.now()
        today = now.date()
        t = now.time().replace(microsecond=0)
        location = get_location()

        # Insert or update attendance row for today
        try:
            # Use UNIQUE(emp_id, date) so insert-ignore then update
            cursor.execute("SELECT * FROM attendance WHERE emp_id=%s AND date=%s", (emp_id, today))
            att = cursor.fetchone()
            if mode == "Login":
                if not att:
                    cursor.execute("INSERT INTO attendance (emp_id, date, login_time, location) VALUES (%s, %s, %s, %s)",
                                   (emp_id, today, t, location))
                else:
                    # if already logged in, do nothing or optionally update location
                    pass
            elif mode == "Logout":
                if att and att['logout_time'] is None:
                    cursor.execute("UPDATE attendance SET logout_time=%s WHERE emp_id=%s AND date=%s",
                                   (t, emp_id, today))
                else:
                    # no login found or already logged out
                    pass
            conn.commit()
        finally:
            cursor.close()
            conn.close()

        return jsonify({"status": "success", "message": f"{mode} marked for {name}", "employee": name}), 200

    else:
        cursor.close()
        conn.close()
        return jsonify({"status": "fail", "message": "Face not recognized"}), 401

# ---------- Monthly Count ----------
@app.route('/monthly_count/<emp_id>', methods=['GET'])
def monthly_count(emp_id):
    # optional query params ?year=2025&month=8
    year = request.args.get('year', datetime.now().year, type=int)
    month = request.args.get('month', datetime.now().month, type=int)

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM attendance 
        WHERE emp_id=%s AND MONTH(date)=%s AND YEAR(date)=%s AND login_time IS NOT NULL
    """, (emp_id, month, year))
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return jsonify({"employee_id": emp_id, "year": year, "month": month, "monthly_attendance": count}), 200

# ---------- Status ----------
@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "running", "message": "API active"}), 200

if __name__ == "__main__":
    app.run(port=9000, debug=True)
