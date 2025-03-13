from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from face_recognition_system import FaceRecognitionSystem

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'known_faces'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize face recognition system
face_system = FaceRecognitionSystem()

@app.route('/')
def index():
    known_faces = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return render_template('index.html', known_faces=known_faces)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    name = request.form.get('name', '').strip()
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not name:
        return jsonify({'error': 'Name is required'}), 400
    
    try:
        # Read the image file
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        # Save and process the face
        filename = f"{name}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, image)
        
        success = face_system.process_new_face(filepath, name)
        
        if success:
            return jsonify({'message': 'Face added successfully'})
        else:
            os.remove(filepath)
            return jsonify({'error': 'Could not detect a clear face in the image'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/recognize', methods=['POST'])
def recognize_face():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read the image file
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        # Process the image and get results
        results = face_system.recognize_face(image)
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/known_faces/<filename>')
def known_faces(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))