import cv2
from PIL import Image
import numpy as np
from picamera2 import Picamera2
import time
from functools import reduce
import motor_control as m
import threading

def sort_in_background(color_name):
    m.color_sort(color_name)

def pixel_to_world(px, py, mm_per_px=0.496, image_width=640, image_height=480):
    cx = image_width // 2
    cy = image_height // 2
    x_mm = (px - cx) * mm_per_px
    y_mm = (cy - py) * mm_per_px
    return x_mm, y_mm

def get_red_mask(hsv_img):
    lower_red1 = np.array([0, 150, 150], dtype=np.uint8)
    upper_red1 = np.array([5, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([175, 160, 160], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    return cv2.bitwise_or(mask1, mask2)

def get_limits(color):
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = hsvC[0][0][0]

    if np.array_equal(color, [0, 255, 255]):  # Yellow
        return np.array([22, 150, 150]), np.array([32, 255, 255])
    elif np.array_equal(color, [0, 255, 0]):  # Green
        return np.array([45, 120, 120]), np.array([75, 255, 255])
    elif np.array_equal(color, [255, 0, 0]):  # Blue
        return np.array([90, 100, 100]), np.array([130, 255, 255])
    else:
        return np.array([hue - 10, 100, 100]), np.array([hue + 10, 255, 255])

def is_in_sorting_zone(x_mm, y_mm):
    SORT_ZONE = {
        "x_min": -20, "x_max": 20,
        "y_min": 70, "y_max": 100
    }
    return (SORT_ZONE["x_min"] <= x_mm <= SORT_ZONE["x_max"] and
            SORT_ZONE["y_min"] <= y_mm <= SORT_ZONE["y_max"])

def run_color_detection():
    red = [0, 0, 255]
    yellow = [0, 255, 255]
    green = [0, 255, 0]
    blue = [255, 0, 0]

    color_bgr_map = {
        "Red": red,
        "Yellow": yellow,
        "Green": green,
        "Blue": blue
    }

    last_sort_time = {color: 0 for color in color_bgr_map}
    cooldown_seconds = 10
    linger_counter = {color: 0 for color in color_bgr_map}
    linger_threshold = 75  # number of consistent frames before sorting
    color_counter = {color: 0 for color in color_bgr_map}
    
    recent_detections = {color: [] for color in color_bgr_map}
    detection_ttl = 3.0  # seconds to remember old detections
    min_distance_mm = 15  # minimum distance to consider it a new cube


    frame_count = 0
    start_time = time.time()

    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(1.0)

    print("Press 'q' to quit.")

    try:
        while True:
            frame_rgb = picam2.capture_array()
            if frame_rgb is None:
                continue

            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            text_strip = np.zeros((120, 640, 3), dtype=np.uint8)
            y_offset = 30
            
            
            frame_color = frame.copy()  # full color
            combined_mask = np.zeros(hsvImage.shape[:2], dtype=np.uint8)  # initialize mask

            for color_name, bgr in color_bgr_map.items():
                if color_name == "Red":
                    mask = get_red_mask(hsvImage)
                else:
                    lower, upper = get_limits(bgr)
                    mask = cv2.inRange(hsvImage, lower, upper)

                combined_mask = cv2.bitwise_or(combined_mask, mask)  # add this mask to the combined



            for color_name, bgr in color_bgr_map.items():
                if color_name == "Red":
                    mask = get_red_mask(hsvImage)
                else:
                    lower, upper = get_limits(bgr)
                    mask = cv2.inRange(hsvImage, lower, upper)
                
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 300:
                        continue

                    x, y, w, h = cv2.boundingRect(cnt)
                    cx = x + w // 2
                    cy = y + h // 2
                    x_mm, y_mm = pixel_to_world(cx, cy)
                    z_mm = 234.95

                    msg = f"{color_name}: ({x_mm:.1f}, {y_mm:.1f}) | Count: {color_counter[color_name]}"
                    cv2.putText(text_strip, msg, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                    y_offset += 25

                    frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)
                    cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    
                    
                    frame_gray = cv2.cvtColor(frame_color.copy(), cv2.COLOR_BGR2GRAY)
                    frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

                    # Restore only the region of interest in grayscale frame
                    frame_gray[y:y+h, x:x+w] = frame_color[y:y+h, x:x+w]
                    
                    # Draw rectangle and label on full color frame
                    cv2.rectangle(frame_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame_color, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


                    current_time = time.time()
                    if is_in_sorting_zone(x_mm, y_mm):
                        linger_counter[color_name] += 1
                    else:
                        linger_counter[color_name] = 0
                    
                    
                    # Remove outdated detections
                    recent_detections[color_name] = [
                        (pos, t) for pos, t in recent_detections[color_name]
                        if current_time - t < detection_ttl
                    ]

                    # Check if this cube is too close to a recently sorted one
                    too_close = any(
                        np.hypot(x_mm - px, y_mm - py) < min_distance_mm
                        for (px, py), _ in recent_detections[color_name]
                    )
                    if too_close:
                        continue

                    
                    
                    if (linger_counter[color_name] >= linger_threshold and
                        current_time - last_sort_time[color_name] > cooldown_seconds):

                        threading.Thread(target=sort_in_background, args=(color_name,), daemon=True).start()
                        last_sort_time[color_name] = current_time
                        linger_counter[color_name] = 0  # reset after triggering
                        color_counter[color_name] += 1  # increment count
                        recent_detections[color_name].append(((x_mm, y_mm), current_time))
                        print(f"Sorted {color_name}. Total: {color_counter[color_name]}")
                        break

            # FPS counter
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 1:
                fps = frame_count / elapsed
                # print(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()

            mask_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)

            color_with_text = np.vstack((frame_color, text_strip))  # combine color frame and text strip
            mask_resized = cv2.resize(mask_bgr, (640, color_with_text.shape[0]))  # resize to match height
            combined_view = np.hstack((color_with_text, mask_resized))  # stack color+text + mask

            cv2.imshow("Combined View", combined_view)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_color_detection()
