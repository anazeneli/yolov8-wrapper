# YOLOv8 Module
A Viam `vision` service for YOLOv8-based object detection with optional tracking capabilities.

## Model `azeneli:yolov8:yolov8`
This model implements the `rdk:service:vision` API for person detection and tracking with configurable zones. It can operate in two modes:

- **Detection-only mode**: Simple frame-by-frame person detection with sequential labeling (person_0, person_1, etc.)
- **Tracking mode**: Maintains persistent object identities across frames with optional zone-based state classification providing persistent tracking IDs and zone-based queue states when tracking mode is enabled.

### Features

- **Real-time Person Detection**: Uses YOLOv8 for accurate person detection (class 0)
- **Dual Operation Modes**:
 - Detection-only: Simple sequential labeling without persistence
 - Tracking: Persistent IDs across frames with optional zone analysis
- **Custom Re-Identification (ReID)**: Support for custom ReID models including OSNet and contrastive learning models
- **Configurable Class Detection**: Filter detection to specific object classes
- **Zone-based State Classification**: Configurable polygon zones for queue management (tracking mode only)
- **Queue State Monitoring**: Tracks person states (WAIT, SUCCESS, ABANDON, WALKING BY) when zones are configured
- **Automatic Mode Selection**: Intelligently chooses detection vs tracking based on configuration
- **Device Optimization**: Supports CUDA, MPS (Apple Silicon), and CPU inference
- **Graceful Fallback**: Automatic fallback to detection-only mode if tracking fails

### Configuration
The following attribute template can be used to configure this model:

```json
{
 "camera_name": "<string>",
 "model_location": "<string>",
 "tracker_config_location": "<string>",
 "classes": ["<class_id>"],
 "zones": {
   "zone_name": [
     [x1, y1],
     [x2, y2],
     [x3, y3]
   ]
 }
}
```

Note: tracker_config_location, classes, and zones are optional attributes.

#### Attributes

| Name | Type | Inclusion | Description |
|------|------|-----------|-------------|
| `camera_name` | string | Required | Name of the camera component providing the video feed |
| `model_location` | string | Optional | Path to the YOLOv8 model weights file (Defaults to `yolov8n.pt`) |
| `tracker_config_location` | string | Optional | Path to the tracker configuration YAML file (e.g., `botsort.yaml`) |
| `classes` | array | Optional | List of class IDs to detect (defaults to all classes) |
| `zones` | object | Optional | Dictionary of zone definitions with polygon coordinates |

### Operating Modes

#### Detection-Only Mode

- **Triggered when**: No `tracker_config_location` provided
- **Output**: Simple detection with class names like `person_0`, `person_1`
- **Use case**: Basic person counting, simple detection applications

#### Tracking Mode

- **Triggered when**: `tracker_config_location` is provided
- **Without zones**: Persistent tracking with class names like `person_123`
- **With zones**: Full queue management with class names like `123_2` (track_id_state)
- **Use case**: Queue analysis, person flow monitoring, zone-based analytics

### Tracker Configuration (YAML)

#### Basic Tracker Config
```yaml
tracker_type: botsort
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 30
match_thresh: 0.8
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: false
```

#### ReID-Enhanced Tracker Config
```yaml
tracker_type: botsort
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 30
match_thresh: 0.8
with_reid: true
model: "/path/to/osnet_reid_model.pt"
proximity_thresh: 0.3
appearance_thresh: 0.4
reid_weight: 0.85
lambda_: 0.95
ema_alpha: 0.8
reid_batch_size: 16
reid_max_distance: 0.4
```

### Zone Configuration

When zones are configured, they must include specific zone names for queue management:

- **WAIT**: Queue waiting area
- **SUCCESS**: Successful completion area  
- **ABANDON**: Abandonment/failure area
- **ENTER**: Entry area

The system maps these zones to state labels:
- WAIT → State 1
- SUCCESS → State 2  
- ABANDON → State 3
- ENTER → Entry tracking
- Default → WALKING BY (State 0)

### Example Configuration

#### Basic Detection (Detection-only mode)
```json
{
 "camera_name": "camera-1",
 "model_location": "/path/to/weights/yolov8n.pt"
}
```

#### Class-Filtered Detection
```json
{
 "camera_name": "camera-1",
 "model_location": "yolov8s.pt",
 "classes": ["0"]
}
```

#### ReID-Enhanced Tracking
```json
{
 "camera_name": "camera-1",
 "model_location": "yolov8m.pt",
 "tracker_config_location": "/path/to/configs/botsort_reid.yaml",
 "classes": ["0"]
}
```

#### Full Tracking with Zones
```json
{
 "camera_name": "camera-1",
 "model_location": "/path/to/weights/yolov8n.pt",
 "tracker_config_location": "/path/to/configs/botsort.yaml",
 "zones": {
   "ABANDON": [
     [406, 582],
     [98, 660],
     [336, 1053],
     [1628, 1062],
     [1767, 945],
     [1374, 770],
     [1106, 843],
     [800, 874],
     [407, 584]
   ],
   "WAIT": [
     [750, 899],
     [755, 791],
     [1108, 773],
     [1142, 897],
     [749, 913]
   ],
   "SUCCESS": [
     [1200, 400],
     [1400, 400], 
     [1400, 600],
     [1200, 600]
   ],
   "ENTER": [
     [100, 100],
     [300, 100],
     [300, 300], 
     [100, 300]
   ]
 }
}
```

### DoCommand

#### `get_current_tracks`
When tracking is enabled, you can retrieve current tracking state:

```python
result = await vision_service.do_command({"command": "get_current_tracks"})
print(result)
```

**Returns current tracks organized by zone state when zones are configured.**

### Output Examples
The `class_name` field in Detection objects varies by mode:

- **Detection-only**: `person_0`, `person_1`, `person_2` (sequential numbering)
- **Tracking without zones**: `person_123`, `person_456` (persistent track IDs)
- **Tracking with zones**: `123_1`, `456_0`, `789_2` (track_id_state_label format)

### Error Handling and Fallback

The module includes robust error handling:

- **Tracking failures**: Automatically falls back to detection-only mode
- **ReID compatibility issues**: Attempts model reload before fallback
- **Invalid configurations**: Validates all parameters during setup
- **Missing files**: Clear error messages for missing model or config files
- **Device optimization**: Automatic selection of best available compute device

### Available Commands

| Command | Description | Returns |
|---------|-------------|---------|
| `get_current_tracks` | Get current tracking state (tracking mode only) | Current tracks by zone |