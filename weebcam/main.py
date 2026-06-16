import cv2
import numpy as np
from tensorflow import keras

# --- Global Variable --- # 
MODEL_PATH = "models/emotion_recognition_model.h5"
IMG_SIZE = 48
EMOTIONS = ['angry', 'happy', 'neutral', 'sad', 'surprise']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
model = keras.models.load_model(MODEL_PATH)

# --- Functions --- #
def DetectFace(frame):
    """ Find faces on a live video frame. """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h),color=(0, 255, 0), thickness=5)
    return frame, faces

def PreprocessFace(frame, x, y, w, h):
    """ Extract and prepare the face for the model. 
        - Convert to greyscale
        - Crop to the detected face
        - Resize to 48x48 (training size)
        - Normalise the pixels between 0 and 1 (rescale=1/255)
        - Add the batch and channel dimensions required by the model
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_roi = gray[y:y + h, x:x + w]
    face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
    face_normalized = face_resized / 255.0
    face_input = face_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    return face_input


def FindEmotionWithModel(frame, faces):
    """ Predict the emotion for every face detected. """
    for (x, y, w, h) in faces:
        face_input = PreprocessFace(frame, x, y, w, h)

        predictions = model.predict(face_input, verbose=0)
        emotion_index = np.argmax(predictions[0])
        emotion_label = EMOTIONS[emotion_index]
        confidence = predictions[0][emotion_index] * 100

        # Show the emotion and % confidence
        print(dict(zip(EMOTIONS, [f"{p*100:.1f}%" for p in predictions[0]])))
        label_text = f"{emotion_label} ({confidence:.1f}%)"
        cv2.putText(
            frame,
            label_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return frame
    
    
# --- Main Part --- #
stream = cv2.VideoCapture(1)
if not stream.isOpened():
    print("No stream :(")
    exit()

fps = stream.get(cv2.CAP_PROP_FPS)
width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
output = cv2.VideoWriter("assets/stream.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize=(width, height))

while True:
    ret, frame = stream.read()
    if not ret:
        print("No more stream :(")
        break

    frame, faces = DetectFace(frame)

    if len(faces) > 0:
        frame = FindEmotionWithModel(frame, faces)

    output.write(frame)
    cv2.imshow("Webcam!", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

stream.release()
output.release()
cv2.destroyAllWindows()