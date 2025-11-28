import cv2


def rover_detection(frame):
    """Detect UGV rovers in the frame based on color."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(
        gray, 50, 255, cv2.THRESH_BINARY_INV
    )  # this works much better
    # _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rovers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:  # min area threshold
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        rovers.append([x + w // 2, y + h // 2, w, h])

    return rovers


def overlap_bboxes(rovers):
    """Remove overlapping bounding boxes."""
    no_overlap = []
    for i, rover in enumerate(rovers):
        x1, y1, w1, h1 = rover
        overlap = False
        for j, rover2 in enumerate(rovers):
            if i != j:
                x2, y2, w2, h2 = rover2
                if (abs(x1 - x2) < (w1 + w2) / 2) and (abs(y1 - y2) < (h1 + h2) / 2):
                    overlap = True
                    # keep the larger bounding box
                    if w1 * h1 < w2 * h2:
                        no_overlap.append(rover2)
                    break
        if not overlap:
            no_overlap.append(rover)
    return no_overlap


test_img_path = (
    "det_loc/det_loc/all_imgs/oak_imgs/8Marker_Detection_screenshot_21.11.2025"
)
if __name__ == "__main__":
    test_img = cv2.imread(test_img_path + ".png")
    detected_rovers = rover_detection(test_img)
    detected_rovers = overlap_bboxes(detected_rovers)

    for rover in detected_rovers:
        x, y, w, h = rover
        cv2.rectangle(
            test_img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2
        )
        cv2.circle(test_img, (x, y), 5, (255, 0, 0), -1)

    cv2.imshow("Detected Rovers", test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
