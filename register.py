import cv2
import os
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import time
import face_recognition
import pickle

dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)
cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
    camera_label.after(10, update_frame)

def register_and_capture():
    name = name_var.get().strip()
    emp_id = id_var.get().strip()

    if not name or not emp_id:
        messagebox.showerror("Input Error", "Please enter both Name and Employee ID")
        return

    folder_name = f"{emp_id}_{name.replace(' ', '_')}"
    folder_path = os.path.join(dataset_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    messagebox.showinfo("Capture Start", "We will take 3 photos (front, left, right). Please get ready!")

    i = 1
    while i <= 3:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Camera Error", "Failed to capture image.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            messagebox.showwarning("Face Not Found", f"No face detected. Please try angle {i} again.")
            time.sleep(1)
            continue

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            filename = os.path.join(folder_path, f"face_{i}.jpg")
            cv2.imwrite(filename, face_img)
            i += 1
            time.sleep(1.5)
            break

    messagebox.showinfo("Success", "All 3 face photos saved successfully!")
    train_model()

def train_model():
    known_encodings = []
    known_names = []

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(folder_name)
            else:
                print(f"[WARNING] No face found in {img_path}")

    data = {"encodings": known_encodings, "names": known_names}
    with open("encodings.pkl", "wb") as f:
        pickle.dump(data, f)
    messagebox.showinfo("Training Complete", "Model trained and encodings saved.")

# GUI setup
root = Tk()
root.title("Attendance System")
root.geometry("900x600")
root.configure(bg="white")

Label(root, text="ATTENDANCE SYSTEM", bg="#2E6DA4", fg="white", font=("Arial", 16, "bold"), width=30, height=2).place(x=0, y=0)

camera_frame = Frame(root, bg="#2E6DA4", bd=5)
camera_frame.place(x=30, y=100, width=500, height=375)
camera_label = Label(camera_frame, bg="white")
camera_label.pack()

form_frame = Frame(root, bg="#2E6DA4", bd=5)
form_frame.place(x=580, y=100, width=280, height=375)
form_inner = Frame(form_frame, bg="white")
form_inner.place(relx=0.05, rely=0.05, relwidth=0.9, relheight=0.9)

Label(form_inner, text="Employee Name:", bg="white", font=("Arial", 12)).pack(pady=(30, 5))
name_var = StringVar()
Entry(form_inner, textvariable=name_var, font=("Arial", 12), width=25).pack()

Label(form_inner, text="Employee ID:", bg="white", font=("Arial", 12)).pack(pady=(20, 5))
id_var = StringVar()
Entry(form_inner, textvariable=id_var, font=("Arial", 12), width=25).pack()

Button(form_inner, text="Register & Capture 3 Photos", bg="green", fg="white", font=("Arial", 12),
       command=register_and_capture).pack(pady=(30, 0))

update_frame()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
