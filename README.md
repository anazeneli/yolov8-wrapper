## Module yolov8 
This module provides YOLOv8-based object detection with optional tracking capabilities.
It can operate in two modes:

- **Detection-only mode**: Simple frame-by-frame person detection with sequential labeling (person_0, person_1, etc.)
- **Tracking mode**: Maintains persistent object identities across frames with optional zone-based state classification



## Model azeneli:yolov8:yolov8
The YOLOv8 model performs person detection and optionally tracks objects across multiple
configurable zones, providing persistent tracking IDs and zone-based queue states when tracking mode is enabled.
Features


## Features

- **Real-time Person Detection**: Uses YOLOv8 for accurate person detection (class 0)
- **Dual Operation Modes**:
  - Detection-only: Simple sequential labeling without persistence
  - Tracking: Persistent IDs across frames with optional zone analysis
- **Zone-based State Classification**: Configurable polygon zones for queue management (tracking mode only)
- **Queue State Monitoring**: Tracks person states (WAIT, SUCCESS, FAILURE, WALKING BY) when zones are configured
- **Automatic Mode Selection**: Intelligently chooses detection vs tracking based on configuration
- **Device Optimization**: Supports CUDA, MPS (Apple Silicon), and CPU inference


## Configuration
The following attribute template can be used to configure this model:

```json
{
  "camera_name": "<string>",
  "model_location": "<string>",
  "tracker_config_location": "<string>",
  "zones": {
    "zone_name": [
      [x1, y1],
      [x2, y2],
      [x3, y3]
    ]
  }
}
```

Note: tracker_config_location and zones are optional attributes.


## Attributes

| Name | Type | Inclusion | Description |
|------|------|-----------|-------------|
| camera_name | string | Required | Name of the camera component providing the video feed |
| tracker_config_location | string | Optional | Path to the tracker configuration YAML file (e.g., botsort.yaml) |
| model_location | string | Optional | Path to the YOLOv8 model weights file (Defaults yolov8.pt) |
| zones | object | Optional | Dictionary of zone definitions with polygon coordinates |


## Operating Modes

### Detection-Only Mode

- **Triggered when: No tracker_config_location provided**
- **Output: Simple detection with class names like person_0, person_1**
Use case: Basic person counting, simple detection applications

### Tracking Mode

- **Triggered when: tracker_config_location is provided**
- **Without zones: Persistent tracking with class names like person_123**
- **With zones: Full queue management with class names like 123_2 (track_id_state)**
  - Use case: Queue analysis, person flow monitoring, zone-based analytics


## Example Configuration
### Basic Detection (Detection-only mode)
```json
{
  "camera_name": "camera-1",
  "model_location": "/path/to/weights/yolov8n.pt"
}
```

### Tracking without Zones
```json
{
  "camera_name": "camera-1",
  "model_location": "/path/to/weights/yolov8n.pt",
  "tracker_config_location": "/path/to/configs/botsort.yaml"
}
```

### Full Tracking with Zones
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
    "ENTER": [
      [750, 899],
      [755, 791],
      [1108, 773],
      [1142, 897],
      [749, 913]
    ]
  }
}
```


## DoCommand

### get_current_tracks
When tracking is enabled, you can retrieve current tracking state:

```python
# Your Python code here
result = await vision_service.do_command({"command": "get_current_tracks"})
print(result)
```

- **Returns current tracks organized by zone state when zones are configured.**

### Output Examples
The `class_name` field in Detection objects varies by mode:

- **Detection-only**: `person_0`, `person_1`, `person_2` (sequential numbering)
- **Tracking without zones**: `person_123`, `person_456` (persistent track IDs)
- **Tracking with zones**: `123_1`, `456_0`, `789_2` (track_id_state_label format)