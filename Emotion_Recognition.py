import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model

# --- STAGE 1: MODEL AND CONFIGURATION LOADING ---
# Path updated to load 'model.h5'.
# Make sure this file is in the same folder as your Python script.
emotion_model = load_model('model.h5')

# Load the face detector
def load_face_detector():
    """Loads the MTCNN face detector."""
    print("INFO: Loading MTCNN face detector...")
    return MTCNN()

# --- STAGE 2: REAL-TIME DETECTION SCRIPT ---

# Define the emotions - Make sure this order matches your model's training
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the models
face_detector = load_face_detector()

print("INFO: Starting webcam feed...")
# Use 0 for the default webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("ERROR: Failed to grab frame.")
        break

    # 1. FACE DETECTION
    detected_faces = face_detector.detect_faces(frame)

    # Process each face found
    for face in detected_faces:
        # Get the bounding box coordinates
        x, y, w, h = face['box']
        
        # Ensure coordinates are within the frame boundaries
        x, y = max(0, x), max(0, y)
        
        # 2. PREPROCESSING FOR EMOTION MODEL
        # Extract the face Region of Interest (ROI)
        face_roi = frame[y:y+h, x:x+w]

        try:
            # === CODE FOR VGG19 MODEL ===
            # The VGG19 model expects a 3-channel (color) image of size 48x48.
            
            # Resize to the target size your model was trained on (48x48)
            resized_face = cv2.resize(face_roi, (48, 48))
            
            # Normalize pixel values to be between 0 and 1
            normalized_face = resized_face / 255.0
            
            # Reshape for the model: Add a batch dimension.
            # The model expects input of shape (1, 48, 48, 3)
            reshaped_face = np.expand_dims(normalized_face, axis=0)

        except Exception as e:
            print(f"Skipping a face due to processing error: {e}")
            continue

        # 3. EMOTION PREDICTION
        # Use the actual trained model to make a prediction
        preds = emotion_model.predict(reshaped_face)[0]
        emotion_index = np.argmax(preds)
        predicted_emotion = EMOTIONS[emotion_index]
        
        # 4. VISUALIZATION
        # Draw the bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Put the emotion text above the bounding box
        text = f"{predicted_emotion}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


    # Display the resulting frame
    cv2.imshow('Emotion Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and destroy windows
video_capture.release()
cv2.destroyAllWindows()
print("INFO: Webcam feed stopped.")

