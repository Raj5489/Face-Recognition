import cv2
import mediapipe as mp
import numpy as np
import os
import face_recognition
import time
from threading import Thread

class FaceRecognitionSystem:
    def __init__(self):
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.7  # Increased confidence threshold
        )

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize known faces
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_metadata = {}  # Store multiple encodings per person
        self.load_known_faces()
        print(f"Loaded {len(self.known_face_names)} known faces: {self.known_face_names}")

        # Performance optimization parameters
        self.frame_count = 0
        self.process_this_frame = True
        self.skip_frames = 3
        self.last_face_locations = []
        self.last_face_names = []

    def load_known_faces(self, faces_dir="known_faces"):
        """Load known faces from directory"""
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
            print(f"Created {faces_dir} directory. Please add face images.")
            return

        print("\nLoading known faces...")
        for filename in os.listdir(faces_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(faces_dir, filename)
                print(f"Processing {filename}...")
                
                try:
                    # Load the image
                    face_image = face_recognition.load_image_file(image_path)
                    
                    # Find face in the image using CNN model for better accuracy
                    face_locations = face_recognition.face_locations(face_image, model="cnn")
                    
                    if len(face_locations) == 0:
                        print(f"No face found in {filename}")
                        continue
                    
                    # Get encoding with high accuracy settings
                    encoding = face_recognition.face_encodings(face_image, [face_locations[0]],
                                                            num_jitters=50,  # Increased for better accuracy
                                                            model="large")[0]  # Use large model
                    
                    name = os.path.splitext(filename)[0]
                    if name not in self.known_face_metadata:
                        self.known_face_metadata[name] = []
                        self.known_face_names.append(name)
                    
                    self.known_face_metadata[name] = [encoding]  # Store single high-quality encoding
                    print(f"Successfully loaded face encoding for {name}")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

        # Flatten encodings for comparison
        self.known_face_encodings = []
        for name in self.known_face_names:
            self.known_face_encodings.extend(self.known_face_metadata[name])

    def process_frame(self, frame):
        if frame is None:
            return None

        # Only process every nth frame to improve performance
        self.frame_count += 1
        if self.frame_count % self.skip_frames != 0:
            if self.last_face_locations:
                return self.draw_results(frame, self.last_face_locations, self.last_face_names)
            return frame

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all faces in the frame using CNN model
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        if face_locations:
            # Get face encodings with increased accuracy
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations,
                                                           num_jitters=2,  # Increase accuracy
                                                           model="large")  # Use large model
            face_names = []

            for face_encoding in face_encodings:
                name = "Unknown"
                if self.known_face_encodings:
                    # Compare with all known face encodings
                    matches = []
                    confidences = {}
                    
                    for known_name in self.known_face_names:
                        # Get all encodings for this person
                        person_encodings = self.known_face_metadata[known_name]
                        
                        # Compare with each encoding
                        person_matches = face_recognition.compare_faces(
                            person_encodings,
                            face_encoding,
                            tolerance=0.5  # Stricter tolerance
                        )
                        
                        if True in person_matches:
                            # Calculate average confidence
                            face_distances = face_recognition.face_distance(
                                person_encodings,
                                face_encoding
                            )
                            avg_confidence = 1 - np.mean(face_distances)
                            confidences[known_name] = avg_confidence
                            matches.extend(person_matches)
                    
                    if matches and True in matches:
                        # Get the name with highest confidence
                        best_match_name = max(confidences.items(), key=lambda x: x[1])[0]
                        confidence = confidences[best_match_name]
                        
                        if confidence > 0.6:  # Higher confidence threshold
                            name = f"{best_match_name} ({confidence:.2%})"
                
                face_names.append(name)

            # Save results for skipped frames
            self.last_face_locations = face_locations
            self.last_face_names = face_names
            
            return self.draw_results(frame, face_locations, face_names)
        
        return frame

    def draw_results(self, frame, face_locations, face_names):
        """Draw the results on the frame"""
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw name below the face with background
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
            cv2.rectangle(frame, (left, bottom - 35), (left + text_size[0] + 10, bottom), 
                         (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                      cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame

    def run_webcam(self):
        print("Initializing webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Failed to open webcam! Trying alternative method...")
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            if not cap.isOpened():
                print("Could not open webcam!")
                return

        print("Webcam opened successfully!")
        last_fps_time = time.time()
        fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Process the frame
                current_time = time.time()
                annotated_frame = self.process_frame(frame)
                
                # Calculate and display FPS every second
                if current_time - last_fps_time >= 1.0:
                    try:
                        fps = 1.0 / (current_time - last_fps_time)
                    except ZeroDivisionError:
                        fps = 0
                    last_fps_time = current_time

                # Display FPS
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display the frame
                cv2.imshow('Face Recognition System', annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Add a small delay to prevent excessive CPU usage
                time.sleep(0.001)

        except Exception as e:
            print(f"Error during webcam operation: {str(e)}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def add_new_face(self, name):
        """Add a new face with high accuracy settings"""
        print("Initializing camera for new face capture...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Failed to open webcam! Trying alternative method...")
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            if not cap.isOpened():
                print("Could not open webcam!")
                return

        print("\nGuidelines for best recognition:")
        print("1. Ensure your face is well-lit")
        print("2. Look directly at the camera")
        print("3. Keep a neutral expression")
        print("4. Position your face in the center of the green box")
        print("5. Maintain a distance of about 2 feet from camera")

        try:
            while True:
                ret, original_frame = cap.read()
                if not ret:
                    continue

                # Create a copy of the frame for displaying guidelines
                display_frame = original_frame.copy()

                # Draw guide box in the center
                h, w = display_frame.shape[:2]
                center_x, center_y = w // 2, h // 2
                box_size = min(w, h) // 2
                top_left = (center_x - box_size // 2, center_y - box_size // 2)
                bottom_right = (center_x + box_size // 2, center_y + box_size // 2)
                
                # Draw guide box and crosshair on display frame only
                cv2.rectangle(display_frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.line(display_frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 1)
                cv2.line(display_frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 1)

                # Real-time face detection feedback
                rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                
                if face_locations:
                    for top, right, bottom, left in face_locations:
                        # Draw face rectangle on display frame only
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (255, 255, 0), 2)
                        
                        # Check if face is centered
                        face_center_x = (left + right) // 2
                        face_center_y = (top + bottom) // 2
                        is_centered = (abs(face_center_x - center_x) < 50 and 
                                     abs(face_center_y - center_y) < 50)
                        
                        if is_centered:
                            cv2.putText(display_frame, "Perfect position! Press 'c' to capture", 
                                      (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            # Show positioning guidance
                            if face_center_x < center_x:
                                cv2.putText(display_frame, "Move right", (10, h-60), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            elif face_center_x > center_x:
                                cv2.putText(display_frame, "Move left", (10, h-60), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            if face_center_y < center_y:
                                cv2.putText(display_frame, "Move down", (10, h-30), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            elif face_center_y > center_y:
                                cv2.putText(display_frame, "Move up", (10, h-30), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, "No face detected", (10, h-30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Show the frame with guidelines
                cv2.imshow('Capture New Face', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    if not face_locations:
                        print("No face detected! Please try again.")
                        continue
                    
                    # Verify face is centered
                    face_center_x = (face_locations[0][3] + face_locations[0][1]) // 2
                    face_center_y = (face_locations[0][0] + face_locations[0][2]) // 2
                    if not (abs(face_center_x - center_x) < 50 and abs(face_center_y - center_y) < 50):
                        print("Face not centered! Please center your face and try again.")
                        continue
                    
                    # Save the original frame without any overlays
                    if not os.path.exists("known_faces"):
                        os.makedirs("known_faces")
                    
                    # Crop and save only the face region
                    top, right, bottom, left = face_locations[0]
                    # Add some padding around the face
                    padding = 50
                    top = max(top - padding, 0)
                    bottom = min(bottom + padding, h)
                    left = max(left - padding, 0)
                    right = min(right + padding, w)
                    face_image = original_frame[top:bottom, left:right]
                    
                    image_path = f"known_faces/{name}.jpg"
                    cv2.imwrite(image_path, face_image)
                    print(f"\nSaved face image for {name}")
                    
                    # Close the capture window
                    cap.release()
                    cv2.destroyAllWindows()
                    
                    # Show processing message
                    print("\nProcessing face encoding with high accuracy settings...")
                    print("This may take a few moments. Please wait...")
                    
                    try:
                        # Process with high accuracy settings but reduced jitters
                        face_image = face_recognition.load_image_file(image_path)
                        face_locations = face_recognition.face_locations(face_image, model="hog")
                        
                        if face_locations:
                            encoding = face_recognition.face_encodings(face_image, face_locations,
                                                                    num_jitters=10,
                                                                    model="large")[0]
                            
                            if name not in self.known_face_metadata:
                                self.known_face_metadata[name] = []
                                self.known_face_names.append(name)
                            
                            self.known_face_metadata[name] = [encoding]
                            self.known_face_encodings = []
                            for n in self.known_face_names:
                                self.known_face_encodings.extend(self.known_face_metadata[n])
                            
                            print("Successfully processed and stored face encoding!")
                            return True
                        else:
                            print("Error: Could not process the saved image. Please try again.")
                            os.remove(image_path)
                            return False
                            
                    except Exception as e:
                        print(f"Error processing face: {str(e)}")
                        if os.path.exists(image_path):
                            os.remove(image_path)
                        return False
                    
                elif key == ord('q'):
                    break

        except Exception as e:
            print(f"Error during face capture: {str(e)}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
        return False

    def process_new_face(self, image_path, name):
        """Process a new face image and add it to known faces"""
        try:
            face_image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(face_image, model="hog")
            
            if not face_locations:
                return False
            
            encoding = face_recognition.face_encodings(
                face_image, 
                [face_locations[0]],
                num_jitters=10,
                model="large"
            )[0]
            
            if name not in self.known_face_metadata:
                self.known_face_metadata[name] = []
                self.known_face_names.append(name)
            
            self.known_face_metadata[name] = [encoding]
            
            # Update flattened encodings
            self.known_face_encodings = []
            for n in self.known_face_names:
                self.known_face_encodings.extend(self.known_face_metadata[n])
            
            return True
            
        except Exception as e:
            print(f"Error processing face: {str(e)}")
            return False

    def recognize_face(self, frame):
        """Recognize faces in an image"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        results = []
        
        if face_locations:
            face_encodings = face_recognition.face_encodings(
                rgb_frame,
                face_locations,
                num_jitters=2,
                model="large"
            )

            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                name = "Unknown"
                confidence = 0
                
                if self.known_face_encodings:
                    matches = []
                    confidences = {}
                    
                    for known_name in self.known_face_names:
                        person_encodings = self.known_face_metadata[known_name]
                        person_matches = face_recognition.compare_faces(
                            person_encodings,
                            face_encoding,
                            tolerance=0.5
                        )
                        
                        if True in person_matches:
                            face_distances = face_recognition.face_distance(
                                person_encodings,
                                face_encoding
                            )
                            avg_confidence = 1 - np.mean(face_distances)
                            confidences[known_name] = avg_confidence
                            matches.extend(person_matches)
                    
                    if matches and True in matches:
                        best_match = max(confidences.items(), key=lambda x: x[1])
                        name = best_match[0]
                        confidence = best_match[1]
                
                results.append({
                    'name': name,
                    'confidence': float(confidence),
                    'location': {
                        'top': int(top),
                        'right': int(right),
                        'bottom': int(bottom),
                        'left': int(left)
                    }
                })
        
        return {'faces': results}

if __name__ == "__main__":
    print("Initializing Face Recognition System...")
    face_system = FaceRecognitionSystem()
    
    while True:
        print("\nFace Recognition System Menu:")
        print("1. Start Face Recognition")
        print("2. Add New Face")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == "1":
            if not face_system.known_face_encodings:
                print("No known faces loaded! Please add faces first.")
                continue
            face_system.run_webcam()
        elif choice == "2":
            name = input("Enter the person's name: ")
            face_system.add_new_face(name)
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.") 