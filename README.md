## Module yolov8-tracker
This module provides YOLOv8-based object detection and tracking with pose estimation capabilities.
It processes camera feeds to detect and track objects across predefined zones,
providing real-time tracking data with zone-based monitoring and transition events.

## Model azeneli:yolov8-tracker:yolov8-tracker
The yolov8-tracker model performs object detection and tracking using YOLOv8 with pose estimation.
It tracks objects across multiple configurable zones and provides tracking IDs,
pose keypoints, and zone transition information for comprehensive object monitoring.

## Configuration
The following attribute template can be used to configure this model:
json{
  "camera_name": "<string>",
  "tracker_config_location": "<string>",
  "model_location": "<string>",
  "zones": {
    "zone_name": [
      [x1, y1],
      [x2, y2],
      [x3, y3]
    ]
  }
}

## Attributes
Name | Type | Inclusion | Description
camera_name | string | Required | Name of the camera component providing the video feed
tracker_config_location | string | Required | Path to the tracker configuration YAML file (e.g., botsort.yaml)
model_location | string | Required | Path to the YOLOv8 model weights file (e.g., yolov8n-pose.pt)
zones | object | Required | Dictionary of zone definitions with polygon coordinates

## Example Configuration
json{
  "camera_name": "camera-1",
  "tracker_config_location": "YOUR/PATH/src/configs/botsort.yaml",
  "model_location": "YOUR/PATH/src/weights/yolov8n-pose.pt",
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
    ],
    "WAIT": [
      [443, 590],
      [908, 480],
      [1405, 674],
      [1097, 811],
      [796, 842],
      [442, 591]
    ],
    "SUCCESS": [
      [889, 473],
      [994, 425],
      [1867, 705],
      [1607, 952],
      [1349, 797],
      [1406, 673],
      [899, 473]
    ]
  }
}
