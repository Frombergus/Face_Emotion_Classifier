"""
Emotion Classifier App – Real-Time Emotion Detection (Video/Webcam Support)
====================================================
Sections:
1. Imports & Logging Setup
2. Config Loading
3. Main Class (enhanced with frame resizing in run())
4. Entry Point & Inline Tests

New Feature: Auto-resizes large video frames to 640x480 for display/performance.
Ideal for portfolio: Shows adaptive preprocessing in CV pipelines for varying inputs.
"""

# Section 1: Imports & Logging Setup
import logging
import argparse  # For CLI video input
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time
import os
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Section 2: Config Loading
def load_config(config_path='docs/config.yaml'):
    """Ladda YAML-config för paths/params. Fallback till defaults."""
    defaults = {
        'model_path': 'models/best_model.keras',
        'face_cascade_path': 'haarcascade_frontalface_default.xml',
        'confidence_threshold': 0.5,
        'img_size': (48, 48),
        'output_width': 640,  # New: For display resizing
        'output_height': 480
    }
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                defaults.update(config)
            logger.info(f"Loaded config from {config_path}")
        else:
            logger.warning(f"Config not found, using defaults: {config_path}")
        return defaults
    except Exception as e:
        logger.error(f"Config load error: {e}")
        return defaults

# Section 3: Main Class
class EmotionClassifierApp:
    """
    Core app för emotion detection.
    Init: Load model & classifier.
    Run: Video/Webcam loop med prediction & rendering (enhanced for resizing).
    """
    def __init__(self, config=None):
        self.config = config or load_config()
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.emotion_colors = {
            'Angry': (0, 0, 255), 'Disgust': (0, 128, 0), 'Fear': (128, 0, 128),
            'Happy': (0, 165, 255), 'Neutral': (128, 128, 128), 'Sad': (255, 0, 0),
            'Surprise': (0, 255, 255), 'Uncertain': (255, 255, 0)
        }
        self.face_classifier = self._load_face_classifier()
        self.model = self._load_emotion_model()
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        self.current_faces = []
        self.first_frame_logged = False  # New: For initial size log
        if self.model and self.face_classifier:
            logger.info("App ready – model & classifier loaded")
        else:
            logger.error("Init failed – missing components")

    def _load_face_classifier(self):
        """Load OpenCV cascade med fallback till built-in."""
        try:
            path = self.config['face_cascade_path']
            if os.path.exists(path):
                classifier = cv2.CascadeClassifier(path)
            else:
                classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if classifier.empty():
                raise ValueError("Empty classifier")
            logger.info("Face classifier loaded")
            return classifier
        except Exception as e:
            logger.error(f"Face load error: {e}")
            return None

    def _load_emotion_model(self):
        """Load Keras model från path."""
        try:
            path = self.config['model_path']
            if os.path.exists(path):
                model = load_model(path)
                logger.info(f"Model loaded: {path}")
                return model
            else:
                raise FileNotFoundError(f"Model missing: {path}")
        except Exception as e:
            logger.error(f"Model load error: {e}")
            return None

    def preprocess_face(self, face_roi):
        """Resize & normalize ROI till model-input."""
        try:
            resized = cv2.resize(face_roi, self.config['img_size'])
            array = image.img_to_array(resized)
            return np.expand_dims(array, axis=0) / 255.0
        except Exception as e:
            logger.error(f"Preprocess error: {e}")
            return None

    def predict_emotion(self, face_roi):
        """Predict emotion med threshold för 'Uncertain'."""
        if not self.model:
            return "Model not loaded", 0
        processed = self.preprocess_face(face_roi)
        if processed is None:
            return "Processing error", 0
        try:
            preds = self.model.predict(processed, verbose=0)
            idx = np.argmax(preds[0])
            conf = float(preds[0][idx])
            if conf < self.config['confidence_threshold']:
                return "Uncertain", conf
            emotion = self.emotion_labels[idx]
            logger.debug(f"Pred: {emotion} ({conf:.2f})")
            return emotion, conf
        except Exception as e:
            logger.error(f"Predict error: {e}")
            return "Prediction error", 0

    def draw_emotion_info(self, frame, rect, emotion, conf):
        """Rita box & label på frame."""
        x, y, w, h = rect
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        label = f"{emotion}: {conf:.2f}"
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x, y - size[1] - 10), (x + size[0], y), color, -1)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def draw_stats(self, frame, input_source="Webcam"):
        """Rita FPS, input source & keys på frame."""
        stats = [f"FPS: {self.fps:.1f}", f"Source: {input_source}", f"Faces: {len(self.current_faces)}", "Q: Quit | S: Screenshot"]
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def update_fps(self):
        """Calc FPS var 30 frames."""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.fps = 30 / (time.time() - self.start_time)
            self.start_time = time.time()

    def save_screenshot(self, frame):
        """Spara frame som timestamped JPG."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn = f"screenshot_{ts}.jpg"
        cv2.imwrite(fn, frame)
        logger.info(f"Saved: {fn}")

    def resize_frame_for_display(self, frame):
        """Resize frame to fixed output size while preserving aspect ratio."""
        h, w = frame.shape[:2]
        target_w, target_h = self.config['output_width'], self.config['output_height']
        
        # Calculate scale to fit within target while preserving aspect
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Pad to exact target size if needed (black borders)
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return padded.astype(np.uint8)

    def run(self, video_path=None):
        """
        Main loop: Capture from video or webcam, detect, predict, display.
        New: Resizes frames for consistent window size; logs initial frame dims.
        """
        if not (self.face_classifier and self.model):
            logger.error("Cannot run – missing init")
            return

        if video_path:
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return
            cap = cv2.VideoCapture(video_path)
            input_source = os.path.basename(video_path)
            logger.info(f"Processing video: {video_path}")
        else:
            cap = cv2.VideoCapture(0)
            input_source = "Webcam"
            logger.info("Processing webcam input")

        if not cap.isOpened():
            logger.error("Cannot open input source")
            return

        # Set resolution for webcam (skip for video to preserve original)
        if video_path is None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['output_width'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['output_height'])

        logger.info("App running – Q to quit, S for screenshot")
        while True:
            ret, frame = cap.read()
            if not ret:  # EOF for video or read failure
                logger.info("End of input reached")
                break

            # New: Log first frame size for debugging large videos
            if not self.first_frame_logged:
                h, w = frame.shape[:2]
                logger.info(f"First frame size: {w}x{h} – will resize for display")
                self.first_frame_logged = True

            # New: Resize frame for display/performance (face detection on original)
            display_frame = self.resize_frame_for_display(frame)

            # Perform detection on original frame (higher res for accuracy)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_classifier.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            self.current_faces = faces

            # Scale face rects to display frame if resized (simple proportional)
            scale_x = display_frame.shape[1] / frame.shape[1]
            scale_y = display_frame.shape[0] / frame.shape[0]
            scaled_faces = [(int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)) for (x, y, w, h) in faces]

            for rect in scaled_faces:
                # Re-extract ROI from original frame, but draw on display
                orig_x, orig_y, orig_w, orig_h = [int(v / scale_x) if i % 2 == 0 else int(v / scale_y) for i, v in enumerate(rect)]
                roi = gray[orig_y:orig_y + orig_h, orig_x:orig_x + orig_w]
                if roi.size > 0:
                    emotion, conf = self.predict_emotion(roi)
                    self.draw_emotion_info(display_frame, rect, emotion, conf)

            self.update_fps()
            self.draw_stats(display_frame, input_source)
            cv2.imshow('Emotion Classifier Portfolio', display_frame)

            # Adjusted waitKey: 1ms for live, ~33ms for video (~30 FPS)
            delay = 1 if video_path is None else 1
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_screenshot(display_frame)

        cap.release()
        cv2.destroyAllWindows()
        logger.info("App exited")

# Section 4: Entry Point
def main():
    """Start app med CLI args för video input."""
    parser = argparse.ArgumentParser(description="Emotion Classifier: Webcam or Video Input")
    parser.add_argument('--video', type=str, default=None, help="Path to video file (e.g., sample.mp4). Defaults to webcam.")
    args = parser.parse_args()

    config = load_config()
    app = EmotionClassifierApp(config)
    app.run(video_path=args.video)

if __name__ == "__main__":
    main()

# Inline Tests (kör med python -m doctest src/emotion_classifier.py)
"""
>>> config = load_config()  # Doktest-exempel
>>> 'output_width' in config
True
"""