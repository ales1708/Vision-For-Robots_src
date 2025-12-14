import heapq
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def heuristic(a, b):
    # We use Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_safe_mask(grid, margin=4):
    mask = grid > 230
    
    for i in range(margin):
        up = np.roll(mask, -1, axis=0)
        down = np.roll(mask, 1, axis=0)
        left = np.roll(mask, -1, axis=1)
        right = np.roll(mask, 1, axis=1)
        
        up[-1, :] = False
        down[0, :] = False
        left[:, -1] = False
        right[:, 0] = False
        
        mask = mask & up & down & left & right
        
    return mask

# To resolve being inside or too close to walls we go to the closest valid position.
def snap_to_safe(mask, point):
    if mask[point]:
        return point 
        
    r, c = point
    height, width = mask.shape
    max_search = 50 
    
    for radius in range(1, max_search):
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                nr, nc = r + i, c + j
                if 0 <= nr < height and 0 <= nc < width:
                    if mask[nr, nc]:
                        return (nr, nc)
    return None

def solve_astar_safe(image_path, safety_margin=4):
    try:
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"Error: Could not find {image_path}.")
        return

    grid = np.array(img)
    height, width = grid.shape

    safe_mask = get_safe_mask(grid, margin=safety_margin)


    plt.imshow(grid, cmap='gray')
    plt.title("Click Start, then Click End")
    plt.axis('off')
    coords = plt.ginput(n=2, timeout=0)
    plt.close()

    if len(coords) < 2:
        return

    raw_start = (int(coords[0][1]), int(coords[0][0]))
    raw_end = (int(coords[1][1]), int(coords[1][0]))

    start = snap_to_safe(safe_mask, raw_start)
    end = snap_to_safe(safe_mask, raw_end)
    
    if start is None or end is None:
        return
        
    start_g = 0
    start_f = heuristic(start, end)
    queue = [(start_f, start_g, start)]
    
    g_scores = {start: 0}
    parents = {start: None}
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
    
    found = False

    while queue:
        current_f, current_g, current_node = heapq.heappop(queue)

        if current_node == end:
            found = True
            break

        if current_g > g_scores.get(current_node, float('inf')):
            continue

        cy, cx = current_node

        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx

            if 0 <= ny < height and 0 <= nx < width:
                if not safe_mask[ny, nx]:
                    continue

                new_g_score = current_g + 1
                
                if new_g_score < g_scores.get((ny, nx), float('inf')):
                    g_scores[(ny, nx)] = new_g_score
                    parents[(ny, nx)] = current_node
                    
                    # A* Priority Calculation
                    new_f_score = new_g_score + heuristic((ny, nx), end)
                    heapq.heappush(queue, (new_f_score, new_g_score, (ny, nx)))

    if not found:
        print("No path found")
        return

    out_img = img.convert("RGB")
    out_pixels = out_img.load()
    
    path_node = end
    while path_node is not None:
        py, px = path_node
        out_pixels[px, py] = (255, 0, 0)
        path_node = parents[path_node]

    for i in range(-2, 3):
        for j in range(-2, 3):
             if 0 <= start[1]+i < width and 0 <= start[0]+j < height:
                 out_pixels[start[1]+i, start[0]+j] = (0, 255, 0)
             if 0 <= end[1]+i < width and 0 <= end[0]+j < height:
                 out_pixels[end[1]+i, end[0]+j] = (0, 0, 255)
    
    plt.imshow(out_img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    solve_astar_safe("maze_combi.pgm", safety_margin=5)