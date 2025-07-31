# Standard library imports
import os
from functools import wraps
from pathlib import Path
from typing import ClassVar, Final, List, Mapping, Optional, Sequence, Tuple, Dict
from urllib.request import urlretrieve

# Third-party imports
import torch
from shapely.geometry import Point, Polygon
from typing_extensions import Self
from ultralytics import YOLO
from ultralytics.engine.results import Results

# Viam imports (third-party framework)
from viam.components.camera import Camera, ViamImage
from viam.logging import getLogger
from viam.media.utils.pil import viam_to_pil_image
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import (Classification, Detection, GetPropertiesResponse)
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import *
from viam.utils import ValueTypes, struct_to_dict

LOGGER = getLogger(__name__)

# Set up decorator for debug logs 
def log_entry(func):
    """A decorator that logs entry into a class method using self.logger."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # The wrapper receives 'self' and can access the instance logger
        self.logger.debug(f"{func.__name__}")
        return func(self, *args, **kwargs)
    return wrapper

def log_all_methods(cls):
    """A class decorator that applies 'log_entry' to all user-defined methods."""
    skip_methods = {'validate_config', 'new', 'check_file_path'}
    
    for attr_name, attr_value in cls.__dict__.items():
        if (callable(attr_value) and 
            not attr_name.startswith("__") and 
            attr_name not in skip_methods):
            setattr(cls, attr_name, log_entry(attr_value))
    return cls

# Global states
LABEL_WALK = 0  #"walking_by"
LABEL_IN = 1 # "queue_in"
LABEL_SUCCESS = 2 #  "queue_success"
LABEL_FAIL = 3 #  "queue_fail"

WAIT = "WAIT"
SUCCESS = "SUCCESS"
FAILURE = "ABANDON"
ENTER = "ENTER"
WALK = "WALKING BY"

# Map zone index to state label
STATES = [WALK, WAIT, SUCCESS, FAILURE]

@log_all_methods
class Yolov8(Vision, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(ModelFamily("azeneli", "yolov8"), "yolov8")

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.camera_name = None
        self.state_labels: dict[int, str] = {} 
        # Return current people in tracks 
        self.current_tracks = {}  # Track ID to state mapping
        self.logger = getLogger(name)

        
    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """This method creates a new instance of this Vision service.
        The default implementation sets the name from the `config` parameter and then calls `reconfigure`.

        Args:
            config (ComponentConfig): The configuration for this resource
            dependencies (Mapping[ResourceName, ResourceBase]): The dependencies (both implicit and explicit)

        Returns:
            Self: The resource
        """
        return super().new(config, dependencies)
    
    
    @classmethod
    def check_file_path(cls, file_path) -> bool:
        """Check if file path exists and is accessible"""
        if not file_path or not isinstance(file_path, str):
            return False
        return os.path.exists(file_path)
    

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """This method allows you to validate the configuration object received from the machine,
        as well as to return any required dependencies or optional dependencies based on that `config`.

        Args:
            config (ComponentConfig): The configuration for this resource

        Returns:
            Tuple[Sequence[str], Sequence[str]]: A tuple where the
                first element is a list of required dependencies and the
                second element is a list of optional dependencies
        """
        optional_dependencies, required_dependencies = [], []
        attrs = struct_to_dict(config.attributes)

        # # Validate required dependencies 
        # Validate camera name
        if "camera_name" not in attrs or not isinstance(attrs["camera_name"], str):
            raise ValueError("camera is required and must be a string")
        required_dependencies.append(attrs["camera_name"])

        # Validate model location or default to YOLOv8n
        try:
            model_location = attrs.get("model_location")
            LOGGER.info(f"Detection enabled")
            
            # Check 1: If model_location not in config, default to yolov8n.pt
            if not model_location or not isinstance(model_location, str):
                LOGGER.info("Model location not specified. Default is yolov8 nano") 
            # Check 2: If it's a YOLO default model, let YOLO handle it automatically
            elif model_location.startswith('yolov8') and model_location.endswith('.pt'):
                LOGGER.info(f"Using YOLO default model: {model_location}")
            # Check 3: Custom model path - must exist locally
            elif not cls.check_file_path(model_location):
                raise FileNotFoundError(f"Custom model file not found at path: {model_location}")
            else:
                LOGGER.info(f"Using custom model: {model_location}")
                
        except FileNotFoundError as e:
            LOGGER.error(f"Model file error: {e}")
            raise
        except Exception as e:
            LOGGER.error(f"Error validating model_location: {e}")
            raise ValueError(f"Invalid model_location configuration: {e}")
                            
        # Optional dependencies 
        # If class_ids_to_use is None, YOLO detects all.
        detection_classes = attrs.get("classes", None)
        if not detection_classes: 
            LOGGER.info(f"Config variable classes not set. Default to all YOLO classes.")
            
        else:
            LOGGER.info(f"Detecting the following classes: {detection_classes}")
            if not isinstance(detection_classes, list):
                raise ValueError("Configuration error: 'detection_classes' must be a list of class names.")
                   
            # Check if all items can be converted to integers
            try:
                [int(item) for item in detection_classes]  # Test conversion
                LOGGER.info(f"Classes validation passed: {detection_classes}")
            except (ValueError, TypeError) as e:
                raise ValueError(f"Configuration error: All items in 'classes' must be numeric strings convertible to integers. Error: {e}")


        try:
            if "tracker_config_location" in attrs:
                LOGGER.info(f"Tracking enabled")
                
                tracker_config = attrs.get("tracker_config_location")
                
                if not isinstance(tracker_config, str):
                    raise TypeError("tracker_config_location must be a string path to your config file")
                
                if not cls.check_file_path(tracker_config):
                    raise FileNotFoundError(f"Tracker config file not found at path: {tracker_config}")

                # Validate zones if tracking is enabled
                if "zones" in attrs:
                    if not isinstance(attrs["zones"], dict):
                        raise TypeError("zones must be a dictionary of polygon list coordinates")
            else:
                # tracker_config_location not provided - detection only mode
                LOGGER.info("Detection only mode - no tracking enabled")
                        
        except FileNotFoundError as e:
            LOGGER.error(f"Tracker config file error: {e}")
            raise
        except TypeError as e:
            LOGGER.error(f"Tracker configuration type error: {e}")
            raise
        except Exception as e:
            LOGGER.error(f"Error validating tracker configuration: {e}")
            raise ValueError(f"Invalid tracker configuration: {e}")
                    

        return required_dependencies, optional_dependencies


    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        print("=== RECONFIGURE STARTING ===")
        attrs = struct_to_dict(config.attributes)
        print(f"Config: {attrs}")

        # Camera 
        camera_component = dependencies.get(Camera.get_resource_name(str(attrs.get("camera_name"))))
        self.camera = camera_component
        
        # Model location with known default 
        model_location = attrs.get("model_location", "yolov8n.pt")
        self.model = YOLO(model_location) 

        # If class_ids_to_use is None, YOLO detects all.
        detection_classes = attrs.get("classes", None)
        if not detection_classes: 
            self.logger.debug(f"Config variable classes not set. Default to all YOLO classes.")
            self.detection_classes = None  # YOLO will detect all classes when None
            self.class_names = self.model.names  # ← Keep as dict: {0: "person", 1: "bicycle", ...}
            self.logger.info(f"No classes specified. Will detect all {len(self.class_names)} classes from model.")

        else:
            detection_classes = [int(item) for item in detection_classes]
            
            # Validate classes exist in this model
            valid_classes = [id for id in detection_classes if id in self.model.names]
            invalid_classes = [id for id in detection_classes if id not in self.model.names]
            
            if invalid_classes:
                self.logger.warning(f"Invalid class IDs {invalid_classes} for this model. Available: {list(self.model.names.keys())}")
                
            self.detection_classes = valid_classes
            
            # Create indexed mapping: class_names[class_id] = "class_name"
            self.class_names = {id: self.model.names[id] for id in valid_classes}
            
            self.logger.info(f"Will detect {len(valid_classes)} specified classes: {list(self.class_names.values())}")
        
        # Only try to set up tracker if tracker_config_location is provided
        tracker_config_location = attrs.get("tracker_config_location", None)
        
        if tracker_config_location:
            try: 
                self.TRACKER_PATH = tracker_config_location
                self.logger.info("Tracker enabled and initialized")
                
                # Only set up zones if tracking is enabled
                if "zones" in attrs: 
                    self.zones = self.prepare_zones(attrs["zones"])
                    self.logger.debug(f"Zones prepared: {self.zones}")
                    # Initialize current tracks
                    self.current_tracks = {zone: [] for zone in self.zones.keys()}
                    self.current_tracks[WALK] = []
                        
            except Exception as e:
                raise Exception(f"Tracker configuration failed: {str(e)}") 
        else:
            self.logger.info("No tracker configured - detection only mode")
    
        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.logger.info(f"Using CUDA device: {self.device}")
        # Check for Mac Metal Performance Shaders (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
            self.logger.info("Using Mac GPU (Metal Performance Shaders)")
        else:
            self.device = "cpu"
            self.logger.info("Using CPU device")

        
        return

    @staticmethod
    def prepare_zones(zones: Dict[str, List]) -> Dict[str, Polygon]:
        """
        Convert zone coordinate data into Shapely Polygon objects.

        Args:
            zones (Dict[str, List]): Dictionary where keys are zone names and values are 
                                    lists of coordinates representing polygon vertices
        Returns:
            Dict[str, Polygon]: Dictionary with Shapely Polygon objects as values
        """
        # convert to numpy arrays
        for zone_name, polygon in zones.items():            
            zones[zone_name] = Polygon(polygon)

        return zones



    def classify_by_feet_centroid(
        self, 
        point: Point, 
        prev_label: int, 
        zones: Dict[str, Polygon]
    ) -> int:
        """
        Classify a person's state based on their foot position and zone containment.
        
        Determines if person is walking by, waiting in queue, succeeded, or failed
        based on which zones contain their ground position.

        Args:
            point (Point): Ground position of the person (foot centroid)
            prev_label (int): Previous state label for context
            zones (dict): Dictionary of zone names to Polygon objects

        Returns:
            int: State label (LABEL_WALK, LABEL_IN, LABEL_SUCCESS, or LABEL_FAIL)
        """        
        # ── Classification Logic ──
        in_queue   = zones[WAIT].contains(point)
        in_fail    = zones[FAILURE].contains(point)
        in_success = zones[SUCCESS].contains(point)
        entered    = zones[ENTER].contains(point) # removed for now

        current_state = LABEL_WALK

        # ── State Classification ──
        if in_queue: 
            current_state = LABEL_IN
        elif in_fail:
            # if prev_label == LABEL_IN or prev_label == LABEL_FAIL:
                # If we were in queue or failed before, we are still in fail
            current_state = LABEL_FAIL
        elif in_success:
            # if prev_label == LABEL_IN or prev_label == LABEL_SUCCESS:
            current_state = LABEL_SUCCESS 

        return current_state


    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> CaptureAllResult:
        """
        Capture image and detections from the configured camera.

        Args:
            camera_name (str): Name of camera (unused, uses configured camera)
            return_image (bool): Whether to return image data
            return_classifications (bool): Whether to return classifications (unused)
            return_detections (bool): Whether to return detections
            return_object_point_clouds (bool): Whether to return point clouds (unused)
            extra: Additional parameters
            timeout: Operation timeout

        Returns:
            CaptureAllResult: Result containing image and detection data
        """        
        result = CaptureAllResult()

        result.image = await self.camera.get_image(mime_type="image/jpeg")
        result.detections = await self.get_detections(result.image)

        return result

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Detection]:
        """
        Get detections from the configured camera.

        Args:
            camera_name (str): Name of camera (unused, uses configured camera)
            extra: Additional parameters
            timeout: Operation timeout

        Returns:
            List[Detection]: List of detected objects with bounding boxes and states
        """        
        image = await self.camera.get_image(mime_type="image/jpeg")

        return await self.get_detections(image)
    
    
    async def get_floor_centroid(
        self, 
        x1: int, 
        y1: int, 
        x2: int, 
        y2: int, 
        track_id: int
    ) -> str:
        """
        Calculate ground position and classify person's state in tracking mode.
        
        Computes the bottom-center of bounding box as foot position, classifies
        the person's state based on zone containment, and updates tracking data.

        Args:
            x1, y1, x2, y2: Bounding box coordinates (top-left and bottom-right)
            track_id: Unique identifier for this tracked person

        Returns:
            str: Queue state string in format "{track_id}_{state_label}"
        """
                
        # Calculate the bottom-center of the bounding box to approximate foot position.
        bottom_center_x = x1 + (x2 - x1) / 2
        bottom_center_y = y2  # The bottom edge of the box
        
        # Create a point representing the person's location on the ground.
        ground_position = Point((bottom_center_x, bottom_center_y))

        # Get the person's previous state for context.
        prev_label = self.state_labels.get(track_id, LABEL_WALK)
        
        # Classify the new point to get the current state.
        state = self.classify_by_feet_centroid(ground_position, prev_label, self.zones)
        
        # Update the state for the current track ID.
        self.state_labels[track_id] = state 
        
        queue_state = f"{str(track_id)}_{str(state)}"
            
        # Update current_tracks
        if hasattr(self, 'current_tracks') and STATES[state] in self.current_tracks:
            self.current_tracks[STATES[state]].append(int(track_id))
        
        
        return queue_state
    

    async def get_detections(
        self, 
        image: ViamImage, 
        *, 
        extra: Optional[Mapping[str, ValueTypes]] = None, 
        timeout: Optional[float] = None
    ) -> List[Detection]:
        """
        Main detection method that routes to appropriate detection mode.
        
        Automatically chooses between tracking mode (with zones and persistent IDs) 
        or detection-only mode (simple person detection) based on configuration.

        Args:
            image (ViamImage): Input image for detection
            extra: Additional parameters
            timeout: Operation timeout

        Returns:
            List[Detection]: List of detected persons. In tracking mode, includes
                            persistent track IDs and queue states. In detection-only 
                            mode, includes simple person labels.
        """
        # Route the task 
        tracker_path = getattr(self, 'TRACKER_PATH', None)

        if tracker_path:
            self.logger.debug(f"Returning tracks" )
            return await self.get_tracks(image, extra=extra, timeout=timeout)

        else:
            self.logger.debug(f"Returning detections only...")
            return await self.get_detections_only(image, extra=extra, timeout=timeout)
                

    async def get_detections_only(
        self, 
        image: ViamImage, 
        *, 
        extra: Optional[Mapping[str, ValueTypes]] = None, 
        timeout: Optional[float] = None
    ) -> List[Detection]:
        """
        Perform basic object detection without tracking or zones.
        
        Uses YOLO's standard detection mode to find people in the image.
        Each detection gets a simple sequential label (person_0, person_1, etc.).
        No persistent tracking IDs or zone-based state classification.

        Args:
            image (ViamImage): Input image for detection
            extra: Additional parameters
            timeout: Operation timeout

        Returns:
            List[Detection]: List of detected persons with class names like 
                            "person_0", "person_1", etc. No tracking or zone states.
        """
        self.logger.info(f"IN GET DETECTIONS ONLY")
        detections = []
        
        try:
            # Convert to PIL image
            pil_image = viam_to_pil_image(image)
            
            # Simple detection (no tracking)
            results = self.model(
                pil_image,                    # Input image for inference
                classes=self.detection_classes,  # Filter to only these class IDs (or None for all)
                device=self.device            # Run on CPU/GPU/MPS
            )[0]  # Get first result from batch (since we're processing 1 image)
            
            if results is None or len(results.boxes) == 0:
                self.logger.debug("No detections found.")
                return detections
            
            self.logger.debug(f"Total detections: {len(results.boxes)}")
            
            # Process detection results                
            for box in results.boxes:
                class_id = int(box.cls.item())
            
                confidence = round(box.conf.item(), 4)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                detection = {
                    "class_name": self.class_names[class_id],  
                    "confidence": confidence,
                    "x_min": x1,
                    "y_min": y1,
                    "x_max": x2,
                    "y_max": y2,
                }
                
                try:
                    detections.append(Detection(**detection))
                except TypeError as e:
                    self.logger.debug(f"Error creating Detection: {str(e)} with data: {detection}")
                    
        except Exception as e:
            self.logger.debug(f"Error in get_detections_only: {str(e)}")
        
        return detections

    async def get_tracks(
        self, 
        image: ViamImage, 
        *, 
        extra: Optional[Mapping[str, ValueTypes]] = None, 
        timeout: Optional[float] = None
    ) -> List[Detection]:
        """
        Perform object detection with persistent tracking and optional zone classification.
        
        Uses YOLO's tracking mode to maintain consistent IDs across frames. If zones
        are configured, applies zone-based state classification for queue management.
        
        Tracking modes:
        - With zones: Returns class names like "123_2" (track_id_state)
        - Without zones: Returns class names like "person_123" (simple track ID)

        Args:
            image (ViamImage): Input image for detection
            extra: Additional parameters
            timeout: Operation timeout

        Returns:
            List[Detection]: List of tracked persons with persistent IDs. If zones
                            are configured, includes queue states (WAIT, SUCCESS, etc.).
                            If no zones, includes simple track IDs.
        """
        detections = []
        
        try:
            # Convert to PIL image
            pil_image = viam_to_pil_image(image)
            
            # Tracking mode
            results = self.model.track(
                pil_image, 
                tracker=self.TRACKER_PATH, 
                persist=True, 
                classes=self.detection_classes, 
                device=self.device
            )[0]
            
            if results is None or len(results.boxes) == 0:
                self.logger.debug("No tracking results found.")
                return detections
            
            # Reset current tracks for each detection call
            if hasattr(self, 'current_tracks'):
                self.current_tracks = {zone: [] for zone in self.current_tracks.keys()}
            
            self.logger.debug(f"Total tracked detections: {len(results.boxes)}")
            
            # Process tracking results
            for i, (xyxy, conf, track_id) in enumerate(
                zip(results.boxes.xyxy, results.boxes.conf, results.boxes.id)
            ):
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                confidence = round(conf.item(), 4)
                track_id = int(track_id.item())
                
                # Only apply zone-based classification if zones exist
                if hasattr(self, 'zones') and self.zones:
                    queue_state = await self.get_floor_centroid(x1, y1, x2, y2, track_id)
                else:
                    # Tracking without zones - just use track ID
                    queue_state = f"person_{track_id}"
                
                detection = {
                    "class_name": queue_state,
                    "confidence": confidence,
                    "x_min": x1,
                    "y_min": y1,
                    "x_max": x2,
                    "y_max": y2,
                }
                
                try:
                    detections.append(Detection(**detection))
                except TypeError as e:
                    self.logger.debug(f"Error creating Detection: {str(e)} with data: {detection}")
                    
        except Exception as e:
            self.logger.debug(f"Error in get_tracks: {str(e)}")
        
        return detections


    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Classification]:
        """
        Get classifications from camera (not implemented).
        
        Args:
            camera_name (str): Camera name
            count (int): Number of classifications to return
            extra: Additional parameters
            timeout: Operation timeout
            
        Returns:
            List[Classification]: Empty list (not implemented)
        """
        self.logger.debug("`get_classifications_from_camera` is not implemented. YOLO is a detection model")
        return [] 

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Classification]:
        """
        Get classifications from image (not implemented).
        
        Args:
            image (ViamImage): Input image
            count (int): Number of classifications to return
            extra: Additional parameters
            timeout: Operation timeout
            
        Returns:
            List[Classification]: Empty list (not implemented)
        """
        self.logger.debug("`get_classifications` is not implemented. YOLO is a detection model")
        return [] 

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[PointCloudObject]:
        """
        Get object point clouds (not implemented).
        
        Args:
            camera_name (str): Camera name
            extra: Additional parameters
            timeout: Operation timeout
            
        Returns:
            List[PointCloudObject]: Empty list (not implemented)
        """
        self.logger.debug("`get_object_point_clouds` is not implemented")
        return [] 

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> Vision.Properties:
        """
        Get the properties/capabilities of this vision service.

        Args:
            extra: Additional parameters
            timeout: Operation timeout

        Returns:
            Vision.Properties: Service capabilities (detections and classifications 
                             supported, point clouds not supported)
        """
        return GetPropertiesResponse(
            classifications_supported=True,
            detections_supported=True,
            object_point_clouds_supported=False
        )
    
    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Mapping[str, ValueTypes]:
        """
        Execute custom commands on the vision service.
        
        Supports 'get_current_tracks' command to retrieve current tracking state
        when tracking is enabled.

        Args:
            command (Mapping[str, ValueTypes]): Command dictionary with 'command' key
            timeout: Operation timeout
            **kwargs: Additional keyword arguments

        Returns:
            Mapping[str, ValueTypes]: Command result or error message
        """
        # Creates object for tracking values at the current timestamp
        if command.get("command") == "get_current_tracks":
            # Log the current state labels
            if not hasattr(self, 'TRACKER_PATH') or not self.TRACKER_PATH:            
                self.logger.debug(f"Tracker is not enabled.")
            else:
                self.logger.debug(f"Current state labels: {self.current_tracks}")
                # Return the current state labels
                return {"current_tracks": self.current_tracks} 

        else:
            return {"error": "Command not recognized"}
        

