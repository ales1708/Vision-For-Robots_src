import cv2
import numpy as np
from PIL import Image
import math

# --- HELPER FUNCTIONS ---
def draw_dashed_line(img, pt1, pt2, color, thickness=1, gap=15, dash_length=8):
    """Helper function to draw dashed lines in OpenCV."""
    pt1 = (int(pt1[0]), int(pt1[1]))
    pt2 = (int(pt2[0]), int(pt2[1]))
    
    dist = math.dist(pt1, pt2)
    if dist < dash_length: 
        cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
        return

    num_dashes = int(dist / gap)
    if num_dashes == 0:
        cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
        return

    for i in range(num_dashes):
        start_t = (i * gap) / dist
        end_t = (i * gap + dash_length) / dist
        
        if end_t > 1.0: end_t = 1.0 

        start_x = int(pt1[0] * (1 - start_t) + pt2[0] * start_t)
        start_y = int(pt1[1] * (1 - start_t) + pt2[1] * start_t)
        end_x = int(pt1[0] * (1 - end_t) + pt2[0] * end_t)
        end_y = int(pt1[1] * (1 - end_t) + pt2[1] * end_t)
        
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness, lineType=cv2.LINE_AA)

def create_connected_pyramid(image_paths, output_name="pyramid_connected.png"):
    # --- CONFIGURATION ---
    vertical_spacing = 100 
    
    layer_target_widths = [600, 450, 300, 150] 
    
    viewing_compression = 0.4  
    perspective_pinch = 0.25   

    if len(image_paths) != 4:
        print("Error: Please provide exactly 4 image paths.")
        return

    layers_data = []

    # 1. PROCESS AND WARP IMAGES
    for i, path in enumerate(image_paths):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: print(f"Error loading {path}"); continue
        if img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        target_w = layer_target_widths[i]
        aspect_ratio = img.shape[0] / img.shape[1]
        target_h = int(target_w * aspect_ratio)
        img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        h, w = img_resized.shape[:2]
        
        src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        squash_h = int(h * viewing_compression)
        pinch_x = int(w * perspective_pinch)
        
        # Order: Top-Left, Top-Right, Bottom-Left, Bottom-Right
        dst_pts = np.float32([
            [pinch_x, 0],            [w - pinch_x, 0], 
            [0, squash_h],           [w, squash_h]
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        warped_img = cv2.warpPerspective(img_resized, M, (w, squash_h), 
                                         flags=cv2.INTER_LINEAR, 
                                         borderMode=cv2.BORDER_CONSTANT, 
                                         borderValue=(0,0,0,0))

        layers_data.append({
            "warped_img": warped_img,
            "corners_relative": dst_pts, 
            "warped_width": w,
            "warped_height": squash_h
        })

    # 2. SETUP CANVAS
    max_warped_w = max(d["warped_width"] for d in layers_data)
    
    raw_total_h = sum(d["warped_height"] for d in layers_data) + \
                  (len(layers_data) * vertical_spacing) + 100
    total_h = int(raw_total_h) 
    
    # Initialize Transparent Canvas
    canvas = np.zeros((total_h, max_warped_w + 200, 4), dtype=np.uint8) 
    
    cx = canvas.shape[1] // 2 
    
    current_y_bottom = total_h - 50 
    
    abs_layer_corners_list = [] 

    # 3. CALCULATE POSITIONS
    for d in layers_data:
        warped_h = d["warped_height"]
        warped_w = d["warped_width"]
        
        x_offset = cx - (warped_w // 2)
        y_offset = current_y_bottom - warped_h
        
        d["canvas_pos"] = (x_offset, y_offset)
        
        layer_abs_corners = []
        for pt in d["corners_relative"]:
            abs_x = int(pt[0] + x_offset)
            abs_y = int(pt[1] + y_offset)
            layer_abs_corners.append((abs_x, abs_y))
        
        abs_layer_corners_list.append(layer_abs_corners)
        
        current_y_bottom -= int(vertical_spacing + warped_h - d["corners_relative"][0][1])

    # 4. DRAW CONNECTING DASHED LINES (Corner to Corner between layers)
    line_color = (150, 150, 150, 255) # Lighter Gray
    
    # Iterate through pairs of layers (0&1, 1&2, 2&3)
    for i in range(len(abs_layer_corners_list) - 1):
        curr_c = abs_layer_corners_list[i]   # Bottom layer in the pair
        next_c = abs_layer_corners_list[i+1] # Top layer in the pair
        
        # Connect corresponding corners
        # 0: Top-Left, 1: Top-Right, 2: Bottom-Left, 3: Bottom-Right
        for j in range(4):
            draw_dashed_line(canvas, curr_c[j], next_c[j], line_color, thickness=2)

    # 5. PASTE IMAGES
    for i, d in enumerate(layers_data):
        x_off, y_off = d["canvas_pos"]
        warped_img = d["warped_img"]
        h_img, w_img = warped_img.shape[:2]
        
        y1, y2 = y_off, y_off + h_img
        x1, x2 = x_off, x_off + w_img

        y1 = max(0, y1); y2 = min(canvas.shape[0], y2)
        x1 = max(0, x1); x2 = min(canvas.shape[1], x2)
        
        img_roi_y1 = max(0, -y_off)
        img_roi_y2 = min(h_img, canvas.shape[0] - y_off)
        img_roi_x1 = max(0, -x_off)
        img_roi_x2 = min(w_img, canvas.shape[1] - x_off)
        
        if y1 < y2 and x1 < x2 and img_roi_y1 < img_roi_y2 and img_roi_x1 < img_roi_x2:
            img_to_blend = warped_img[img_roi_y1:img_roi_y2, img_roi_x1:img_roi_x2]
            
            canvas_roi = canvas[y1:y2, x1:x2]
            alpha_s = img_to_blend[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            
            for c in range(0, 3):
                canvas_roi[:, :, c] = (alpha_s * img_to_blend[:, :, c] +
                                       alpha_l * canvas_roi[:, :, c])
            canvas_roi[:, :, 3] = np.maximum(canvas_roi[:, :, 3], img_to_blend[:, :, 3])

    # Save and Show
    cv2.imwrite(output_name, canvas)
    print(f"Saved {output_name}")
    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)).show()


# --- HELPER TO GENERATE DUMMY IMAGES ---
def get_dummy_images():
    names = []
    base_colors = [(100, 100, 255), (100, 255, 100), (255, 150, 50), (255, 50, 50)] 
    
    for i in range(4):
        size = 600 
        img = np.full((size, size, 3), (25, 25, 25), dtype=np.uint8) 
        
        grid_spacing = 50
        for r in range(0, size, grid_spacing):
            for c in range(0, size, grid_spacing):
                if (r // grid_spacing + c // grid_spacing) % 2 == 0:
                     cv2.rectangle(img, (c, r), (c + grid_spacing, r + grid_spacing), base_colors[i], -1)

        for j in range(0, size, grid_spacing):
            cv2.line(img, (j, 0), (j, size), (200, 200, 200), 1) 
            cv2.line(img, (0, j), (size, j), (200, 200, 200), 1)
        
        fname = f"input_L{4-i}.png" 
        cv2.imwrite(fname, img)
        names.append(fname)
    return names

# --- RUN ---
if __name__ == "__main__":
    # Generate and use dummy files if real ones aren't present
    dummy_files = get_dummy_images() 
    input_files = [f"field_zon19.png", f"img_1.5.png", f"img_3.5.png", f"img_5.0.png"]

    create_connected_pyramid(input_files)