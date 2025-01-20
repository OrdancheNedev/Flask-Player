import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from flask import Flask, Response, render_template, request, redirect, url_for, send_file
import time
import zipfile

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/video'
IMAGE_FOLDER = 'uploads/images'
RECOGNIZED_FOLDER = 'uploads/recognized'
ARCHIVE_FOLDER = 'uploads/archive'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['RECOGNIZED_FOLDER'] = RECOGNIZED_FOLDER
app.config['ARCHIVE_FOLDER'] = ARCHIVE_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(RECOGNIZED_FOLDER, exist_ok=True)
os.makedirs(ARCHIVE_FOLDER, exist_ok=True)

def clear_previous_files():
    for folder in [UPLOAD_FOLDER, IMAGE_FOLDER, RECOGNIZED_FOLDER, ARCHIVE_FOLDER]:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        clear_previous_files()

        if 'video' not in request.files or 'image' not in request.files:
            return "No file part", 400

        video_file = request.files['video']
        image_file = request.files['image'] 

        if video_file.filename == '' or image_file.filename == '':
            return "No selected file", 400

        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        image_path = os.path.join(app.config['IMAGE_FOLDER'], image_file.filename)

        video_file.save(video_path)
        image_file.save(image_path)

        return redirect(url_for('play_video', filename=video_file.filename))

    return render_template('index.html')

@app.route('/play/<filename>')
def play_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return render_template('play.html', video_url=url_for('video_feed', filename=filename))

def generate_frames(video_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(keep_all=True, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    uploaded_images = os.listdir(app.config['IMAGE_FOLDER'])
    if not uploaded_images:
        print("No images found in the image folder.")
        return

    reference_image_path = os.path.join(app.config['IMAGE_FOLDER'], uploaded_images[0])
    reference_image = cv2.imread(reference_image_path)
    reference_image_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
    reference_faces = mtcnn(reference_image_rgb)

    if reference_faces is None:
        print("No face found in the reference image.")
        return

    reference_embedding = model(reference_faces[0].unsqueeze(0).to(device)).detach().cpu().numpy()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1 / fps if fps > 0 else 0.033
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = mtcnn(rgb_frame)

        if faces is not None:
            for i, face in enumerate(faces):
                face_embedding = model(face.unsqueeze(0).to(device)).detach().cpu().numpy()
                distance = np.linalg.norm(reference_embedding - face_embedding)
                threshold = 0.7

                if distance < threshold:
                    bbox = mtcnn.detect(rgb_frame)[0][i]
                    left, top, right, bottom = bbox.astype(int)
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, 'Recognized!', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    recognized_frame_path = os.path.join(app.config['RECOGNIZED_FOLDER'], f'recognized_{frame_count}.jpg')
                    cv2.imwrite(recognized_frame_path, frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(delay)

    cap.release()

@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download_recognized')
def download_recognized():
    zip_path = 'uploads/archive/recognized_frames.zip'
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(app.config['RECOGNIZED_FOLDER']):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, app.config['RECOGNIZED_FOLDER']))
    return send_file(zip_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
