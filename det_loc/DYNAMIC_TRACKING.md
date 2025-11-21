# Dynamic Tag Tracking for Improved Localization

## Overview

The dynamic tag tracking system automatically pans the camera to keep the two closest AprilTags in view while the robot is driving. This significantly improves localization accuracy by ensuring the most relevant tags are always visible.

## How It Works

### 1. **Finding Closest Tags**

```python
from det_loc.utils import find_closest_tags

# Returns the 2 closest tags with their distances
closest_tags = find_closest_tags(
    robot_position=[2.5, 3.0, 0.5],  # [x, y, rotation]
    num_tags=2
)
# Returns: [(tag_id, distance, [x, y]), ...]
# Example: [(4, 1.2, [4.5, 0.0]), (5, 1.5, [4.5, 6.0])]
```

**Algorithm:**
1. Calculate Euclidean distance from robot to each tag
2. Sort tags by distance
3. Return top N closest tags

### 2. **Calculating Pan Angle**

```python
from det_loc.utils import calculate_pan_to_tags

# Calculate pan angle to center the tags
pan_angle = calculate_pan_to_tags(
    robot_position=[2.5, 3.0, 0.5],  # [x, y, rotation in radians]
    closest_tags=closest_tags
)
# Returns: pan angle in radians (e.g., 0.3 rad ≈ 17°)
```

**Algorithm:**
1. Calculate centroid (average position) of closest tags
2. Calculate angle from robot to centroid (world frame)
3. Convert to pan angle relative to robot heading
4. Normalize to [-π, π]

**Formula:**
```
pan_angle = arctan2(dy, dx) - robot_heading
```
where `dx`, `dy` are offsets from robot to tag centroid.

### 3. **Dynamic Tracking Class**

```python
from det_loc.utils import DynamicTagTracker

# Initialize tracker
tracker = DynamicTagTracker(
    camera_controller=camera,
    config={
        "num_tags_to_track": 2,
        "update_threshold_rad": 0.1,  # Update if change > 0.1 rad (~5.7°)
        "smoothing_factor": 0.3,       # Exponential smoothing weight
    }
)

# Enable tracking
tracker.enable()

# Update camera pan based on robot position
tracker.update(robot_position=[2.5, 3.0, 0.5], logger=logger)
```

**Features:**
- **Automatic updates**: Continuously adjusts camera pan
- **Smoothing**: Exponential smoothing prevents jerky movements
- **Threshold filtering**: Only updates when change is significant
- **Boundary clamping**: Respects camera pan limits

## Integration with Main Node

The system is automatically integrated into `ImageSubscriber`:

### Initialization
```python
node = ImageSubscriber(
    enable_localization=True,
    enable_navigation=True
)
```

### Automatic Activation
1. **During initial scan**: Camera performs sweep to find tags
2. **After scan locks**: Dynamic tracking is automatically enabled
3. **While driving**: Camera continuously tracks closest tags

### Workflow

```
┌─────────────────┐
│  Start Robot    │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Camera Scanning │  ← Find initial tags
│  (10 seconds)   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Lock Camera on  │  ← Lock to best initial view
│   Best View     │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Enable Dynamic  │  ← Start tracking
│   Tracking      │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ While Driving:  │
│ 1. Detect tags  │
│ 2. Localize     │
│ 3. Find closest │
│ 4. Pan camera   │  ← Continuous loop
│ 5. Navigate     │
└─────────────────┘
```

## Configuration Parameters

### `DynamicTagTracker` Config

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_tags_to_track` | int | 2 | Number of closest tags to track |
| `update_threshold_rad` | float | 0.1 | Min angle change to trigger update (radians) |
| `camera_offset` | float | 0.0 | Camera mounting offset from robot heading |
| `smoothing_factor` | float | 0.3 | Exponential smoothing weight (0-1) |

### Smoothing Explained

Exponential moving average prevents jerky camera movements:

```
new_pan = α × desired_pan + (1-α) × old_pan
```

- `α = 0.3` (default): 30% new, 70% old → smooth tracking
- `α = 1.0`: No smoothing → instant response
- `α = 0.0`: No updates → fixed pan

## Usage Examples

### Example 1: Basic Dynamic Tracking

```python
from det_loc.utils import DynamicTagTracker, find_closest_tags

# Setup
tracker = DynamicTagTracker(camera_controller)
tracker.enable()

# In control loop
while driving:
    robot_pos = get_robot_position()  # [x, y, rotation]
    tracker.update(robot_pos)
    
    # Get tracking info
    info = tracker.get_tracking_info()
    print(f"Tracking tags: {info['tracked_tags']}")
    print(f"Camera pan: {info['current_pan_degrees']:.1f}°")
```

### Example 2: Manual Tag Selection

```python
from det_loc.utils import find_closest_tags, calculate_pan_to_tags

robot_pos = [2.5, 3.0, 0.5]

# Find 3 closest tags
closest = find_closest_tags(robot_pos, num_tags=3)

# Use only first 2 for panning
pan_angle = calculate_pan_to_tags(robot_pos, closest[:2])

# Manually set camera
camera.set_pan(pan_angle)
```

### Example 3: Custom Tag Positions

```python
# Define custom tag map
custom_tags = {
    "tag1": [10.0, 5.0],
    "tag2": [8.0, 2.0],
    "tag3": [5.0, 8.0],
}

# Find closest with custom map
closest = find_closest_tags(
    robot_position=[6.0, 4.0, 0.0],
    tag_positions=custom_tags,
    num_tags=2
)
```

## Benefits

### 1. **Improved Localization Accuracy**
- Always views the most relevant tags
- Reduces triangulation error
- Better distance measurements

### 2. **Robust to Motion**
- Compensates for robot rotation
- Maintains tag visibility while turning
- Smooth camera transitions

### 3. **Optimized for Field Layout**
- Adapts to robot position on field
- Automatically switches between tag pairs
- Works with any tag configuration

## Performance Characteristics

### Computational Cost
- **Tag distance calculation**: O(N) where N = number of tags (10 in standard field)
- **Sorting**: O(N log N) - negligible for small N
- **Pan calculation**: O(1)
- **Total per update**: < 1ms on typical hardware

### Update Frequency
- **Default threshold**: 0.1 rad (~5.7°)
- **Typical update rate**: 1-2 Hz during active driving
- **Zero updates** when stationary or moving straight

### Camera Movement
- **Smoothing factor 0.3**: ~3-5 updates to reach target
- **Prevents oscillation**: Threshold filtering
- **Respects limits**: Auto-clamps to [-1.5, +1.5] rad

## Debugging

### Enable Verbose Logging

```python
# Tracker will log every pan update
tracker.update(robot_pos, logger=node.get_logger())
```

**Sample output:**
```
[INFO] Tracking tags [4, 5]: pan=0.32 rad (18.3°)
[INFO] Tag 4 distance: 1.234 m
[INFO] Tag 5 distance: 1.456 m
```

### Get Tracking Status

```python
info = tracker.get_tracking_info()
print(f"Enabled: {info['enabled']}")
print(f"Tracked tags: {[t['id'] for t in info['tracked_tags']]}")
print(f"Distances: {[t['distance'] for t in info['tracked_tags']]}")
print(f"Current pan: {info['current_pan_degrees']:.1f}°")
```

### Disable Tracking

```python
tracker.disable()  # Camera stops automatic updates
```

## Advanced Topics

### Camera Offset Calibration

If your camera is mounted at an angle:

```python
# Camera mounted 15° to the right
tracker_config = {
    "camera_offset": np.radians(15)  # 0.262 radians
}
```

### Adaptive Thresholds

Adjust update sensitivity based on speed:

```python
# Fast movement → more sensitive
if robot_speed > 0.3:
    tracker.update_threshold_rad = 0.05  # 2.9°
else:
    tracker.update_threshold_rad = 0.1   # 5.7°
```

### Multi-Tag Tracking

Track more than 2 tags for redundancy:

```python
tracker_config = {
    "num_tags_to_track": 3  # Use 3 closest tags
}
```

## Troubleshooting

### Issue: Camera oscillates
**Cause**: Smoothing factor too high  
**Solution**: Reduce `smoothing_factor` to 0.2 or lower

### Issue: Camera response too slow
**Cause**: Smoothing factor too low or threshold too high  
**Solution**: Increase `smoothing_factor` or decrease `update_threshold_rad`

### Issue: No tags in view
**Cause**: Robot too far from all tags  
**Solution**: Check APRILTAG_POSITIONS match your field setup

### Issue: Wrong tags being tracked
**Cause**: Incorrect robot position estimate  
**Solution**: Verify triangulation is working correctly

## Testing

```python
# Unit test for closest tags
def test_find_closest():
    robot_pos = [4.5, 3.0, 0.0]
    closest = find_closest_tags(robot_pos, num_tags=2)
    
    assert len(closest) == 2
    assert closest[0][1] < closest[1][1]  # Sorted by distance
    print(f"Closest tags: {[t[0] for t in closest]}")

# Unit test for pan calculation
def test_pan_calculation():
    robot_pos = [2.0, 3.0, 0.0]
    closest = find_closest_tags(robot_pos, num_tags=2)
    pan = calculate_pan_to_tags(robot_pos, closest)
    
    assert -np.pi <= pan <= np.pi
    print(f"Pan angle: {np.degrees(pan):.1f}°")
```

## References

- AprilTag detection: `tag_detector.py`
- Camera control: `camera_controller.py`
- Localization: `localization.py`
- Main integration: `detection_subscriber.py`

