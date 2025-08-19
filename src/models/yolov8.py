
# ================================================================================================
# YOLO Vision Service for Viam Robotics Platform
# ================================================================================================
# A comprehensive computer vision service that provides:
# - Object detection using YOLOv8
# - Multi-object tracking with persistent IDs (BoTSORT/ByteTrack)
# - Re-identification (ReID) for robust tracking
# - Zone-based state classification for queue management
# - GPU acceleration support (CUDA, Apple MPS)
# ================================================================================================

import os
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import ClassVar, Final, List, Mapping, Optional, Sequence, Tuple, Dict
from urllib.request import urlretrieve
import io
from contextlib import redirect_stdout, redirect_stderr

# Third-party imports
from shapely.geometry import Point, Polygon
import torch
from typing_extensions import Self
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils import LOGGER as ultra_logger
import yaml

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
 
# Suppress YOLO logging 
ultra_logger.disabled = True

 
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
            
        # Configuration flags (set during reconfigure)
        self.tracking_enabled = False
        self.zones_enabled = False
        self.reid_enabled = False

        
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
        """Check if file path exists and is accessible (local files only)"""
        if not file_path or not isinstance(file_path, str):
            return False
        return os.path.exists(file_path)

    @classmethod
    def parse_yolo_yaml(cls, yaml_path: str) -> Optional[str]:
        """
        Parse a YOLO YAML configuration file to extract the model path.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            str: Model path if found, None otherwise
        """
        try:
            with open(yaml_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
            
            # Common keys where model path might be specified in YOLO configs
            model_keys = ['model', 'path', 'weights', 'model_path', 'pt_path']
            
            for key in model_keys:
                if key in yaml_content:
                    model_path = yaml_content[key]
                    if model_path and isinstance(model_path, str):
                        LOGGER.info(f"Found model path in YAML under '{key}': {model_path}")
                        return model_path
            
            LOGGER.warning(f"No model path found in YAML file: {yaml_path}")
            return None
            
        except yaml.YAMLError as e:
            LOGGER.error(f"Invalid YAML format in file {yaml_path}: {e}")
            return None
        except Exception as e:
            LOGGER.error(f"Error reading YAML file {yaml_path}: {e}")
            return None

    @classmethod
    def _parse_tracker_config(cls, config_path: str) -> dict:
        """Parse tracker YAML configuration file and resolve ReID model paths"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if not config:
                return {}
            
            # If ReID is enabled and has a model path, resolve it
            if config.get('with_reid', False) and config.get('model'):
                original_path = config['model']
                try:
                    resolved_path = cls.resolve_model_path(original_path, "reid")
                    config['model'] = resolved_path
                    LOGGER.info(f"Resolved ReID model path: {original_path} -> {resolved_path}")
                except FileNotFoundError as e:
                    LOGGER.error(f"ReID model path resolution failed: {e}")
                    raise
            
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in tracker config: {e}")
        except Exception as e:
            raise ValueError(f"Error reading tracker config file: {e}")

    
    @classmethod
    def is_yolo_default_model(cls, model_path: str) -> bool:
        """Check if model path is a YOLO default model"""
        if not model_path or not isinstance(model_path, str):
            return False
        return model_path.startswith('yolov8') and model_path.endswith('.pt')
    
    @classmethod
    def is_local_file(cls, file_path: str) -> bool:
        """Check if file path exists locally"""
        if not file_path or not isinstance(file_path, str):
            return False
        return os.path.exists(file_path)
    
    @classmethod
    def is_yaml_file(cls, file_path: str) -> bool:
        """Check if file path is a YAML file"""
        if not file_path or not isinstance(file_path, str):
            return False
        return file_path.lower().endswith(('.yaml', '.yml'))
    
    @classmethod
    def resolve_model_path(cls, model_location: str, context: str = "detection") -> str:
        """
        Resolve model path from various sources and apply routing logic.
        
        Args:
            model_location: The model location from config (could be path, YAML, etc.)
            context: Context for the model ("detection" or "reid")
            
        Returns:
            str: Resolved model path
            
        Raises:
            FileNotFoundError: If custom model path cannot be resolved and no fallback available
        """
        if not model_location or not isinstance(model_location, str):
            if context == "detection":
                LOGGER.info("Model location not specified. Defaulting to yolov8n.pt")
                return "yolov8n.pt"
            else:
                raise FileNotFoundError(f"Model location required for {context} but not specified")
        
        # If it's a YAML file, try to extract model path from it
        if cls.is_yaml_file(model_location):
            if cls.is_local_file(model_location):
                LOGGER.info(f"Parsing YAML config file: {model_location}")
                extracted_path = cls.parse_yolo_yaml(model_location)
                if extracted_path:
                    # Recursively resolve the extracted path
                    return cls.resolve_model_path(extracted_path, context)
                else:
                    if context == "detection":
                        LOGGER.warning(f"Could not extract model path from YAML {model_location}, using default")
                        return "yolov8n.pt"
                    else:
                        raise FileNotFoundError(f"Could not extract model path from YAML {model_location}")
            else:
                if context == "detection":
                    LOGGER.warning(f"YAML file not found: {model_location}, using default")
                    return "yolov8n.pt"
                else:
                    raise FileNotFoundError(f"YAML file not found: {model_location}")
        
        # Check if it's a YOLO default model
        if cls.is_yolo_default_model(model_location):
            LOGGER.info(f"Using YOLO default model: {model_location}")
            return model_location
        
        # Check if it's a local file
        if cls.is_local_file(model_location):
            LOGGER.info(f"Using local model file: {model_location}")
            return model_location
        
        # File not found - apply context-specific fallback
        if context == "detection":
            LOGGER.warning(f"Model file not found: {model_location}, using default yolov8n.pt")
            return "yolov8n.pt"
        else:
            raise FileNotFoundError(f"Custom {context} model file not found: {model_location}")
    
    @classmethod
    def validate_file_path(cls, file_path: str, file_type: str = "file") -> tuple[bool, str]:
        """
        Validate file path and return validation result with description
        
        Args:
            file_path: Path to validate
            file_type: Type of file for error messages (e.g., "model", "tracker config")
            
        Returns:
            tuple: (is_valid: bool, description: str)
        """
        if not file_path or not isinstance(file_path, str):
            return False, f"{file_type} path is empty or invalid"
        
        if cls.is_yolo_default_model(file_path):
            return True, f"YOLO default {file_type}"
        
        if cls.is_local_file(file_path):
            return True, f"Local {file_type}"
        
        return False, f"Local {file_type} not found at path: {file_path}"
    
    @classmethod
    def validate_tracker_config(cls, tracker_config: str) -> dict:
        """Validate tracker config and return parsed ReID settings"""
        if not isinstance(tracker_config, str):
            raise TypeError("tracker_config_location must be a string path to your config file")
        
        is_valid, description = cls.validate_file_path(tracker_config, "tracker config")
        
        if not is_valid:
            raise FileNotFoundError(description)
        
        LOGGER.info(f"Using {description}: {tracker_config}")
        
        # Only parse local files
        if cls.is_local_file(tracker_config):
            return cls._parse_tracker_config(tracker_config)
        
        return {}
    
    def _detect_and_set_device(self) -> str:
        """
        Detect and set the best available device for inference.
        
        Priority order:
        1. CUDA (NVIDIA GPU) - if available
        2. MPS (Apple Silicon GPU) - if available  
        3. CPU - fallback
        
        Returns:
            str: The device identifier that was set
        """
        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            self.device = f"cuda:{device_id}"
            device_name = torch.cuda.get_device_name(device_id)
            self.logger.info(f"Using CUDA device {self.device}: {device_name}")
        
        # Check for Mac Metal Performance Shaders (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
            self.logger.info("Using Mac GPU (Metal Performance Shaders)")
        
        # Fallback to CPU
        else:
            self.device = "cpu"
            self.logger.info("Using CPU device")

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

        # Validate required dependencies 
        # Validate camera name
        if "camera_name" not in attrs or not isinstance(attrs["camera_name"], str):
            raise ValueError("camera is required and must be a string")
        required_dependencies.append(attrs["camera_name"])

        # Validate model location with new resolution logic
        try:
            model_location = attrs.get("model_location")
            LOGGER.info(f"Detection enabled")
            
            # Use the new resolve_model_path method
            resolved_model = cls.resolve_model_path(model_location, "detection")
            LOGGER.info(f"Model validation passed: {resolved_model}")
                
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

        # Validate tracking configuration 
        try:
            if "tracker_config_location" in attrs:
                LOGGER.info(f"Tracking configuration found")
                
                tracker_config = attrs.get("tracker_config_location")
                
                if not isinstance(tracker_config, str):
                    raise TypeError("tracker_config_location must be a string path to your config file")
                
                if not cls.check_file_path(tracker_config):
                    raise FileNotFoundError(f"Tracker config file not found at path: {tracker_config}")
                else:
                    LOGGER.info(f"Using local tracker config: {tracker_config}")
                    
                    # Parse tracker config for ReID settings with new resolution logic
                    reid_config = cls._parse_tracker_config(tracker_config)
                    if reid_config.get('with_reid', False) and reid_config.get('model'):
                        # ReID model path is already resolved in _parse_tracker_config
                        LOGGER.info("ReID model validation passed")
                    else:
                        LOGGER.info("Tracking enabled without ReID")
            else:
                LOGGER.info("No tracking configuration - detection only mode")
                        
        except FileNotFoundError as e:
            LOGGER.error(f"Tracker config file error: {e}")
            raise
        except TypeError as e:
            LOGGER.error(f"Tracker configuration type error: {e}")
            raise
        except Exception as e:
            LOGGER.error(f"Error validating tracker configuration: {e}")
            raise ValueError(f"Invalid tracker configuration: {e}")
                        
        # Validate zones configuration (independent of tracking)
        try:
            if "zones" in attrs:
                LOGGER.info(f"Zone configuration found")
                if not isinstance(attrs["zones"], dict):
                    raise TypeError("zones must be a dictionary of polygon list coordinates")
                LOGGER.info(f"Zones validation passed: {list(attrs['zones'].keys())}")
            else:
                LOGGER.info("No zone configuration")
                        
        except TypeError as e:
            LOGGER.error(f"Zone configuration type error: {e}")
            raise
        except Exception as e:
            LOGGER.error(f"Error validating zone configuration: {e}")
            raise ValueError(f"Invalid zone configuration: {e}")
        
        return required_dependencies, optional_dependencies


    def reconfigure(
            self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
        ) -> None:
            attrs = struct_to_dict(config.attributes)

            # Camera 
            camera_component = dependencies.get(Camera.get_resource_name(str(attrs.get("camera_name"))))
            self.camera = camera_component
            
            # Model location with new resolution logic
            model_location = attrs.get("model_location")
            resolved_model_path = self.resolve_model_path(model_location, "detection")
            self.model = YOLO(resolved_model_path, verbose=False) # Change verbose to true for robust logging 

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
            
            # Initialize ReID configuration
            self.reid_enabled = False
            self.reid_config = {}
            
            # Only try to set up tracker if tracker_config_location is provided
            tracker_config_location = attrs.get("tracker_config_location", None)

            if tracker_config_location:
                try: 
                    self.TRACKER_PATH = tracker_config_location
                    self.logger.info("Tracker enabled and initialized")
                    self.tracking_enabled = True

                    # Parse ReID configuration from tracker config
                    try:
                        self.reid_config = self._parse_tracker_config(tracker_config_location)
                        
                        # Check if ReID is enabled
                        if self.reid_config.get('with_reid', False):
                            reid_model_path = self.reid_config.get('model')
                            if reid_model_path:
                                self.reid_enabled = True
                                self.reid_model_path = reid_model_path
                                
                                # Log ReID configuration details
                                reid_settings = {
                                    'proximity_thresh': self.reid_config.get('proximity_thresh', 0.3),
                                    'appearance_thresh': self.reid_config.get('appearance_thresh', 0.4),
                                    'reid_weight': self.reid_config.get('reid_weight', 0.85),
                                    'lambda_': self.reid_config.get('lambda_', 0.95),
                                    'ema_alpha': self.reid_config.get('ema_alpha', 0.8),
                                    'reid_batch_size': self.reid_config.get('reid_batch_size', 16),
                                    'reid_max_distance': self.reid_config.get('reid_max_distance', 0.4)
                                }
                                
                                self.logger.info(f"ReID enabled with model: {reid_model_path}")
                                self.logger.debug(f"ReID settings: {reid_settings}")
                            else:
                                self.logger.warning("ReID enabled in config but no model path specified")
                        else:
                            self.logger.info("ReID disabled in tracker configuration")
                            
                    except Exception as e:
                        self.logger.warning(f"Could not parse ReID config from tracker file: {e}")
                        self.logger.info("Continuing with tracking without ReID configuration parsing")
                            
                except Exception as e:
                    raise Exception(f"Tracker configuration failed: {str(e)}") 
            else:
                self.logger.info("No tracker configured - detection only mode")
                self.tracking_enabled = False

            # Set up zones if configured (independent of tracking)
            if "zones" in attrs: 
                self.zones = self.prepare_zones(attrs["zones"])
                self.logger.debug(f"Zones prepared: {self.zones}")
                self.zones_enabled = True
                # Initialize current tracks for zones
                self.current_tracks = {zone: [] for zone in self.zones.keys()}
                self.current_tracks[WALK] = []
            else:
                self.logger.debug("No zones configured")
                self.zones_enabled = False
        
            # Check for CUDA (NVIDIA GPU)
            self._detect_and_set_device()



    def prepare_zones(self, zones_config: Dict[str, List]) -> Dict[str, Polygon]:
        """
        Convert zone coordinate data into Shapely Polygon objects.

        Args:
            zones_config (Dict[str, List]): Dictionary where keys are zone names and values are 
                                       lists of coordinates representing polygon vertices
        Returns:
            Dict[str, Polygon]: Dictionary with Shapely Polygon objects as values
        """
        # Convert to Polygon arrays
        prepared_zones = {}
        for zone_name, polygon_coords in zones_config.items():            
            prepared_zones[zone_name] = Polygon(polygon_coords)

        return prepared_zones



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
        detections = []
        
        try:
            # Convert to PIL image
            pil_image = viam_to_pil_image(image)
            
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
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
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
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
        
        Supported commands:
        - 'get_current_tracks': Retrieve current tracking state
        - 'get_config_status': Get configuration status (what's enabled/disabled)

        Args:
            command (Mapping[str, ValueTypes]): Command dictionary with 'command' key
            timeout: Operation timeout
            **kwargs: Additional keyword arguments

        Returns:
            Mapping[str, ValueTypes]: Command result or error message
        """
        
        cmd = command.get("command", "").lower()
        
        if cmd == "get_current_tracks":
            if not self.tracking_enabled:            
                self.logger.debug("Tracker is not enabled.")
                return {"current_tracks": {}, "tracker_enabled": False}
            else:
                self.logger.debug(f"Current state labels: {self.current_tracks}")
                return {
                    "current_tracks": self.current_tracks,
                    "tracker_enabled": True,
                    "state_labels": getattr(self, 'state_labels', {})
                }
        
        elif cmd == "get_config_status":
            return {
                "timestamp": datetime.now().isoformat(),
                "features_enabled": {
                    "tracking": self.tracking_enabled,
                    "reid": self.reid_enabled,
                    "zones": self.zones_enabled
                },
                "model_info": {
                    "device": getattr(self, 'device', 'unknown'),
                    "model_configured": hasattr(self, 'model') and self.model is not None,
                    "camera_configured": hasattr(self, 'camera') and self.camera is not None,
                    "total_classes": len(getattr(self, 'class_names', {}))
                },
                "reid_details": {
                    "enabled": self.reid_enabled,
                    "model_path": getattr(self, 'reid_model_path', None),
                    "settings_count": len(getattr(self, 'reid_config', {}))
                } if self.reid_enabled else {"enabled": False},
                "zones_details": {
                    "enabled": self.zones_enabled,
                    "count": len(getattr(self, 'zones', {})),
                    "zone_names": list(getattr(self, 'zones', {}).keys())
                } if self.zones_enabled else {"enabled": False}
            }
        
        else:
            return {
                "error": f"Command '{cmd}' not recognized", 
                "available_commands": ["get_current_tracks", "get_config_status"]
            }