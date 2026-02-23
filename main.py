import cv2
import easyocr
import sys
import numpy as np

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
        self.tracked_items = []
        self.opencv_version = cv2.__version__

    def detect_colors(self, frame, threshold_percentage=10.0):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width, _ = frame.shape
        total_pixels = height * width
        detected_colors = []

        # Further adjusted HSV color ranges for better yellow detection
        color_ranges = {
            "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
            "green": [((36, 100, 100), (86, 255, 255))],
            "yellow": [((20, 80, 80), (40, 255, 255))]
        }

        for color_name, ranges in color_ranges.items():
            mask = cv2.inRange(hsv_frame, np.array(ranges[0][0]), np.array(ranges[0][1]))
            if len(ranges) > 1:
                mask2 = cv2.inRange(hsv_frame, np.array(ranges[1][0]), np.array(ranges[1][1]))
                mask = cv2.bitwise_or(mask, mask2)

            percentage = (cv2.countNonZero(mask) / total_pixels) * 100
            if percentage > threshold_percentage:
                detected_colors.append((color_name, percentage))
        
        return detected_colors

    def get_tracked_items(self):
        ret, frame = self.cap.read()
        if not ret:
            return [], []

        # Detect significant colors in the frame
        detected_colors = self.detect_colors(frame)
        
        # Convert frame to RGB for correct color display for character ROIs
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
            self.tracked_items = []

            for (bbox, text, prob) in results:
                if len(text) == 1 and prob > 0.1 and len(self.trackers) < self.MAX_TRACKERS:
                    (top_left, top_right, bottom_right, bottom_left) = bbox
                    x_coords = [p[0] for p in [top_left, top_right, bottom_right, bottom_left]]
                    y_coords = [p[1] for p in [top_left, top_right, bottom_right, bottom_left]]
                    
                    x, y = int(min(x_coords)), int(min(y_coords))
                    w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                    
                    if w > 0 and h > 0 and x >= 0 and y >= 0 and (x + w) <= frame_width and (y + h) <= frame_height:
                        # Calculate average color of the bounding box from the RGB frame
                        roi = rgb_frame[y:y+h, x:x+w]
                        avg_color = cv2.mean(roi)[:3]

                        tracker = create_tracker(self.opencv_version)
                        # Initialize tracker with the original BGR frame
                        tracker.init(frame, (x, y, w, h))
                        self.trackers.append(tracker)
                        self.tracked_items.append((text, avg_color))
        else:
            updated_trackers = []
            updated_items = []
            for i, tracker in enumerate(self.trackers):
                # Update tracker with the original BGR frame
                success, _ = tracker.update(frame)
                if success:
                    updated_trackers.append(tracker)
                    updated_items.append(self.tracked_items[i])
            
            self.trackers = updated_trackers
            self.tracked_items = updated_items

        return self.tracked_items, detected_colors

    def release(self):
        self.cap.release()

def main():
    detector = CharacterDetector()
    try:
        for i in range(500):
            items, detected_colors = detector.get_tracked_items()
            if items:
                # avg_color is (R, G, B)
                formatted_items = [f"{char} (Color: R={int(r)}, G={int(g)}, B={int(b)})" for char, (r, g, b) in items]
                print(f"Frame {i}: Detected items: {', '.join(formatted_items)}")
            elif detected_colors:
                formatted_colors = [f"{name} ({percentage:.2f}%)" for name, percentage in detected_colors]
                print(f"Frame {i}: No character detected. Detected colors: {', '.join(formatted_colors)}")
    except KeyboardInterrupt:
        print("\nStopping detection.")
    finally:
        detector.release()
        print("Camera released.")

if __name__ == "__main__":
    main()
