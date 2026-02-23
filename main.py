import cv2
import easyocr
import sys

def create_tracker(opencv_version):
    """Creates a tracker object based on the OpenCV version."""
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


def main():
    """
    Main function to run a performant and robust real-time OCR for 
    detecting and tracking SINGLE CHARACTERS.
    """
    cap = cv2.VideoCapture(0)
    
    # --- READER CONFIGURATION ---
    reader = easyocr.Reader(['en'], gpu=False, quantize=True)

    # --- Core Parameters ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592 / 6)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944 / 6)
    OCR_FRAME_INTERVAL = 20
    MAX_TRACKERS = 5
    
    # --- ALLOWLIST: Define the specific characters you want to detect ---
    # This improves accuracy and performance by filtering out unwanted characters.
    ALLOWLIST_CHARS = 'HSU'

    # --- State Variables ---
    frame_counter = 0
    trackers = []
    tracked_texts = []
    opencv_version = cv2.__version__

    while True:
        ret, frame = cap.read()
        display = frame.copy()

        if not ret:
            break

        frame_height, frame_width, _ = frame.shape
        frame_counter += 1

        if frame_counter >= OCR_FRAME_INTERVAL:
            frame_counter = 0

            # --- Image Pre-processing for OCR ---
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            results = reader.readtext(
                processed_frame,  # Use the processed frame for detection
                link_threshold=0.1,
                min_size=10,
                allowlist=ALLOWLIST_CHARS,
            )

            trackers = []
            tracked_texts = []

            for (bbox, text, prob) in results:
                # --- Stricter Confidence and Single Character Check ---
                if len(text) == 1 and prob > 0.1 and len(trackers) < MAX_TRACKERS:
                    (top_left, top_right, bottom_right, bottom_left) = bbox
                    x_coords = [p[0] for p in [top_left, top_right, bottom_right, bottom_left]]
                    y_coords = [p[1] for p in [top_left, top_right, bottom_right, bottom_left]]
                    
                    x, y = int(min(x_coords)), int(min(y_coords))
                    w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                    
                    if w > 0 and h > 0 and x >= 0 and y >= 0 and (x + w) <= frame_width and (y + h) <= frame_height:
                        tracker = create_tracker(opencv_version)
                        tracker.init(frame, (x, y, w, h))
                        trackers.append(tracker)
                        tracked_texts.append(text)

        else:
            for i, tracker in enumerate(trackers):
                success, box = tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display, tracked_texts[i], (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Real-time Character Recognition', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
