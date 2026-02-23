import cv2
import easyocr
import sys

def create_tracker(opencv_version):
    (major, minor, _) = opencv_version.split('.')

    if int(major) >= 4 and int(minor) >= 5:
        try:
            return cv2.legacy.TrackerKCF_create()
        except AttributeError:
            print("\n[ERROR] Your OpenCV version requires the 'legacy' module.")
            print("The 'KCF' tracker is not in the main 'opencv-python' package.")
            print("Please run: pip install opencv-contrib-python\n")
            sys.exit()
    else:
        try:
            return cv2.TrackerKCF_create()
        except AttributeError:
            print("\n[ERROR] 'TrackerKCF_create' not found.")
            print("Your OpenCV version might be too old or missing modules.")
            print("Consider running: pip install opencv-contrib-python\n")
            sys.exit()


class CharacterDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
        self.reader = easyocr.Reader(['en'], gpu=False, quantize=True)

        # Optimized resolution for speedy detection with no gpu
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592 / 6)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944 / 6)

        self.OCR_FRAME_INTERVAL = 40
        self.MAX_TRACKERS = 5

        # All possible chars in the maze
        self.ALLOWLIST_CHARS = 'HSU'

        self.frame_counter = 0
        self.trackers = []
        self.tracked_texts = []
        self.opencv_version = cv2.__version__

    def get_tracked_characters(self):
        ret, frame = self.cap.read()
        if not ret:
            return []

        frame_height, frame_width, _ = frame.shape
        self.frame_counter += 1

        if self.frame_counter >= self.OCR_FRAME_INTERVAL:
            self.frame_counter = 0

            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = self.reader.readtext(
                processed_frame,
                link_threshold=0.1,
                min_size=10,
                allowlist=self.ALLOWLIST_CHARS,
            )

            self.trackers = []
            self.tracked_texts = []

            for (bbox, text, prob) in results:
                if len(text) == 1 and prob > 0.1 and len(self.trackers) < self.MAX_TRACKERS:
                    (top_left, top_right, bottom_right, bottom_left) = bbox
                    x_coords = [p[0] for p in [top_left, top_right, bottom_right, bottom_left]]
                    y_coords = [p[1] for p in [top_left, top_right, bottom_right, bottom_left]]
                    
                    x, y = int(min(x_coords)), int(min(y_coords))
                    w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                    
                    if w > 0 and h > 0 and x >= 0 and y >= 0 and (x + w) <= frame_width and (y + h) <= frame_height:
                        tracker = create_tracker(self.opencv_version)
                        tracker.init(frame, (x, y, w, h))
                        self.trackers.append(tracker)
                        self.tracked_texts.append(text)
        else:
            for i, tracker in enumerate(self.trackers):
                success, _ = tracker.update(frame)
                if not success:
                    self.trackers.pop(i)
                    self.tracked_texts.pop(i)

        return self.tracked_texts

    def release(self):
        self.cap.release()

def main():
    detector = CharacterDetector()
    try:
        for i in range(500):
            characters = detector.get_tracked_characters()
            if characters:
                print(f"Frame {i}: Currently tracked characters: {', '.join(characters)}")
    except KeyboardInterrupt:
        print("\nStopping detection.")
    finally:
        detector.release()
        print("Camera released.")

if __name__ == "__main__":
    main()
