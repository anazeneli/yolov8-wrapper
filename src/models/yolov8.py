"""
YOLOv8 Vision Service with Custom ReID Tracking

Viam component that provides:
- Object detection using YOLOv8 models (CPU/CUDA/MPS optimized)  
- Multi-object tracking with custom BOTSort + OSNet ReID
- Zone-based queue state classification for retail/service applications
- Graceful fallback between tracking and detection-only modes
- Comprehensive error handling and logging

Key Features:
- Custom BOTSort tracker with enhanced OSNet ReID model
- Automatic device optimization (CUDA > MPS > CPU)
- Zone-based people state tracking (WAIT, SUCCESS, FAILURE, WALK)
- Robust error handling with detection-only fallback
- Comprehensive logging and monitoring capabilities

Configuration:
- camera_name: Required camera dependency
- model_location: YOLO model path (defaults to yolov8n.pt)
- classes: Optional list of class IDs to detect
- tracker_config_location: Optional BOTSort config for tracking
- zones: Optional polygon zones for state classification
"""

# Standard library imports
import io
import logging
import os
import time
import warnings
from contextlib import redirect_stdout, redirect_stderr
from functools import wraps
from pathlib import Path
from typing import ClassVar, Final, List, Mapping, Optional, Sequence, Tuple, Dict
from urllib.request import urlretrieve

# Third-party imports
import torch
import yaml
from shapely.geometry import Point, Polygon
from typing_extensions import Self, Any

# Ultralytics imports
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.trackers import track
from ultralytics.utils import LOGGER as ultra_logger

# Custom tracker imports
from tracker.bot_sort import BOTSort

# Viam framework imports
from viam.components.camera import Camera, ViamImage
from viam.logging import getLogger
from viam.media.utils.pil import viam_to_pil_image
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection, GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import *
from viam.utils import ValueTypes, struct_to_dict

# Configure logging and warnings
os.environ['YOLO_VERBOSE'] = 'False'
warnings.filterwarnings("ignore")
ultra_logger.disabled = True

# Replace ultralytics BOTSort with custom implementation
track.TRACKER_MAP['botsort'] = BOTSort

# Initialize logger
LOGGER = getLogger(__name__)
  
# Set up decorator for debug logs 
def log_entry(func):
    """A decorator that logs entry into a class method using self.logger."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
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
    MODEL: ClassVar[Model] = Model(ModelFamily("azeneli", "yolov8-wrapper"), "yolov8")

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
        """
        Checks if a file path is valid and accessible.

        This method verifies that the input is a non-empty string and that a
        file or directory exists at that path on the filesystem.

        Args:
            file_path (Any): The file path to check.

        Returns:
            bool: `True` if the path exists, `False` otherwise.
        """
        if not file_path or not isinstance(file_path, str):
            return False
        return os.path.exists(file_path)
    
    @classmethod
    def _validate_camera(cls, attrs: Dict[str, Any], required_dependencies: list):
        """
        Validates the required 'camera_name' attribute.

        Ensures that 'camera_name' exists in the configuration, is a non-empty
        string, and adds it to the list of required dependencies for the component.

        Args:
            attrs (Dict[str, Any]): The component's configuration attributes.
            required_dependencies (list): The list to which the camera name will be
                appended if validation is successful.

        Returns:
            None: This method does not return a value.

        Raises:
            ValueError: If 'camera_name' is missing or is not a string.
        """
        camera_name = attrs.get("camera_name")
        if not camera_name or not isinstance(camera_name, str):
            raise ValueError("'camera_name' is required and must be a string.")
        required_dependencies.append(camera_name)
        LOGGER.info(f"Validated camera dependency: {camera_name}")

    @classmethod
    def _validate_model(cls, attrs: Dict[str, Any]):
        """
        Validates the 'model_location' attribute.

        Checks the provided model location. If not specified, it logs a default.
        If a custom path is provided, it verifies the file exists.

        Args:
            attrs (Dict[str, Any]): The component's configuration attributes.

        Returns:
            None: This method does not return a value.

        Raises:
            TypeError: If 'model_location' is provided but is not a string.
            FileNotFoundError: If 'model_location' points to a custom model
                that does not exist.
        """
        model_location = attrs.get("model_location")

        # Case 1: Not provided. Default to yolov8n.pt.
        if not model_location:
            LOGGER.info("No 'model_location' specified. Defaulting to yolov8n.pt.")
            return

        # Case 2: Provided, but not a string. Invalid.
        if not isinstance(model_location, str):
            raise TypeError("'model_location' must be a string.")

        # Case 3: A default YOLO model.
        if model_location.startswith('yolov8') and model_location.endswith('.pt'):
            LOGGER.info(f"Using YOLO default model: {model_location}")
        # Case 4: A custom model path. Must exist.
        elif not cls.check_file_path(model_location):
            raise FileNotFoundError(f"Custom model file not found at path: {model_location}")
        else:
            LOGGER.info(f"Using custom model: {model_location}")

    @classmethod
    def _validate_classes(cls, attrs: Dict[str, Any]):
        """
        Validates the optional 'classes' attribute.

        Ensures that if the 'classes' attribute is provided, it is a list of
        numeric strings that can be successfully converted to integers.

        Args:
            attrs (Dict[str, Any]): The component's configuration attributes.

        Returns:
            None: This method does not return a value.

        Raises:
            TypeError: If 'classes' is not a list.
            ValueError: If any item in the 'classes' list cannot be converted
                to an integer.
        """
        detection_classes = attrs.get("classes")

        if detection_classes is None:
            LOGGER.info("No 'classes' specified. Defaulting to all model classes.")
            return

        if not isinstance(detection_classes, list):
            raise TypeError("'classes' must be a list of numeric strings.")

        try:
            # Ensure all items in the list are convertible to integers
            [int(item) for item in detection_classes]
            LOGGER.info(f"Detecting the following classes: {detection_classes}")
        except (ValueError, TypeError) as e:
            raise ValueError(
                "All items in 'classes' must be numeric strings (e.g., [\"0\", \"2\"]). "
                f"Error: {e}"
            )
    
    @classmethod
    def _validate_zones(cls, attrs: Dict[str, Any]):
        """
        Validates the optional 'zones' attribute.

        Ensures that if a 'zones' attribute is provided, it is structured
        as a dictionary.

        Args:
            attrs (Dict[str, Any]): The component's configuration attributes.

        Returns:
            None: This method does not return a value.

        Raises:
            TypeError: If 'zones' is not a dictionary.
        """
        zones = attrs.get("zones")

        if zones is None:
            # Zones are optional, so we just return if they're not present.
            return

        if not isinstance(zones, dict):
            raise TypeError("'zones' must be a dictionary of polygon coordinates.")
        
        LOGGER.info("Zone configuration found and validated.")

    def _setup_device(self) -> None:
        """
        Sets the optimal compute device for PyTorch operations.

        It checks for available hardware in the order of NVIDIA CUDA, then
        Apple Metal Performance Shaders (MPS), and finally defaults to CPU.
        Uses device strings to avoid GPU synchronization overhead.
        """
        if torch.cuda.is_available():
            self.device = "cuda"  # Use string to avoid GPU sync
            self.logger.info(f"Using CUDA device: {self.device}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
            self.logger.info("Using Mac GPU (Metal Performance Shaders)")
        else:
            self.device = "cpu"
            self.logger.info("Using CPU device")

    def _reconfigure_model(self, attrs: Dict) -> None:
        """
        Reconfigures the YOLO model if its location has changed.

        Compares the new model location from the attributes with the current one.
        If they differ, it loads a new model instance. If no location is
        provided, it defaults to 'yolov8n.pt'.

        Args:
            attrs (Dict): The component's configuration attributes.
        """
        model_location = attrs.get("model_location", "yolov8n.pt")
        if model_location != getattr(self, 'model_location', None):
            self.logger.info(f"Loading model from: {model_location}")
            try:
                self.model = YOLO(model_location, verbose=False)
                self.model_location = model_location
                # Add validation that model actually loaded
                self.logger.info(f"Model loaded successfully. Available classes: {list(self.model.names.keys())}")
            except Exception as e:
                self.logger.error(f"Failed to load model from {model_location}: {e}")
                raise
            
    def _check_model_state(self):
        """Debug method to check model state"""
        if not hasattr(self, 'model') or self.model is None:
            self.logger.error("Model not loaded")
            return False
        
        if not hasattr(self, 'class_names') or self.class_names is None:
            self.logger.error("Class names not configured")
            return False
            
        self.logger.debug(f"Model loaded: {type(self.model)}")
        self.logger.debug(f"Available classes: {list(self.class_names.keys())}")
        return True

    def _reconfigure_classes(self, attrs: Dict) -> None:
        """
        Reconfigures and validates the list of object detection classes.

        Based on the 'classes' attribute, this method sets which object classes
        to detect. It validates any provided class IDs against the loaded
        model's capabilities and defaults to all classes if none are specified.

        Args:
            attrs (Dict): The component's configuration attributes.
        """
        if not hasattr(self, 'model') or self.model is None:
            self.logger.warning("Model not loaded, cannot configure classes.")
            return

        detection_classes = attrs.get("classes")
        if not detection_classes:
            self.logger.info("No classes specified. Will detect all classes from the model.")
            self.detection_classes = None  # A value of None means "detect all"
            self.class_names = self.model.names
        else:
            class_ids = [int(item) for item in detection_classes]

            # Validate requested class IDs against the model's actual classes
            valid_classes = [id for id in class_ids if id in self.model.names]
            invalid_classes = [id for id in class_ids if id not in self.model.names]
            
            if invalid_classes:
                self.logger.warning(f"Invalid class IDs {invalid_classes} for this model. Available: {list(self.model.names.keys())}")
            
            self.detection_classes = valid_classes
            self.class_names = {id: self.model.names[id] for id in valid_classes}
            self.logger.info(f"Will detect {len(valid_classes)} specified classes: {list(self.class_names.values())}")
    
    def _reconfigure_tracker(self, attrs: Dict) -> None:
        """
        Simple tracker configuration that ensures device gets passed correctly to BOTSort.
        
        Validates tracker configuration file and prepares for ultralytics integration.
        The actual BOTSort instance will be created automatically by ultralytics
        with the device parameter passed through.
        """
        try:
            tracker_config_location = attrs.get("tracker_config_location")
            
            if tracker_config_location:
                self.logger.info(f"Configuring tracker: {tracker_config_location}")
                
                # Validate config file exists
                if not os.path.exists(tracker_config_location):
                    raise FileNotFoundError(f"Tracker config not found: {tracker_config_location}")
                
                # Load and validate YAML
                with open(tracker_config_location, 'r') as f:
                    config = yaml.safe_load(f)
                    
                if not isinstance(config, dict):
                    raise ValueError("Tracker config must be a dictionary")
                
                self.tracker_config_location = tracker_config_location
                
                # Log configuration status
                self.logger.info(f"✓ Tracker configured - device will be: {self.device}")
                if config.get('with_reid', False):
                    reid_model = config.get('model', 'auto')
                    self.logger.info(f"✓ ReID enabled: {reid_model}")
                else:
                    self.logger.info("✓ Basic tracking (no ReID)")
                    
            else:
                self.tracker_config_location = None
                self.logger.info("Detection-only mode (no tracker)")
                
        except Exception as e:
            self.logger.error(f"Tracker configuration error: {e}")
            self.logger.info("Falling back to detection-only mode")
            self.tracker_config_location = None

    def _reconfigure_zones(self, attrs: Dict) -> None:
        """
        Reconfigures detection zones and initializes their tracking state.

        This method processes the 'zones' attribute, calling `prepare_zones` to
        convert raw coordinates into Polygon objects, and then sets up the
        necessary state for tracking objects within these new zones.

        Args:
            attrs (Dict): The component's configuration attributes.

        Raises:
            Exception: If the zone configuration is malformed and fails preparation.
        """
        try:
            if "zones" in attrs:
                self.zones = self.prepare_zones(attrs["zones"])
                self.logger.debug(f"Zones prepared: {self.zones}")
                # Initialize current tracks state
                self.current_tracks = {zone: [] for zone in self.zones.keys()}
            else:
                self.zones = {}
                self.current_tracks = {}
        except Exception as e:
            raise Exception(f"Zone configuration failed: {str(e)}")

    def prepare_zones(self, zones: Dict[str, List]) -> Dict[str, Polygon]:
        """
        Converts raw zone coordinate data into Shapely Polygon objects.

        Args:
            zones (Dict[str, List]): A dictionary where keys are zone names
                and values are lists of [x, y] vertex coordinates.

        Returns:
            Dict[str, Polygon]: A dictionary of zone names to their
                corresponding Shapely Polygon objects.
        """
        prepared_zones = {}
        
        for zone_name, polygon_coords in zones.items():
            prepared_zones[zone_name] = Polygon(polygon_coords)
            
        return prepared_zones

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """
        Validates the configuration, separating logic for each variable
        and ensuring dependencies between them are checked explicitly.
        """
        optional_dependencies, required_dependencies = [], []
        attrs = struct_to_dict(config.attributes)

        # Validate each configuration variable using a dedicated helper method
        cls._validate_camera(attrs, required_dependencies)
        cls._validate_model(attrs)
        cls._validate_classes(attrs)
        # Remove tracker validation from class method - needs instance logger
        cls._validate_zones(attrs)

        return required_dependencies, optional_dependencies

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        """
        Dynamically reconfigures the component based on a new configuration.

        This method serves as an orchestrator, calling individual helper methods
        to update the component's state for the model, classes, tracker, and
        zones. It assumes the configuration has already been validated.

        Args:
            config (ComponentConfig): The new configuration for this resource.
        """
        self.logger.info("Reconfiguring the component...")
        attrs = struct_to_dict(config.attributes)

        # Camera 
        camera_component = dependencies.get(Camera.get_resource_name(str(attrs.get("camera_name"))))
        self.camera = camera_component

        self._setup_device()
        
        # Reconfigure Detection Model
        self._reconfigure_model(attrs)
        self._reconfigure_classes(attrs)  # This must run after the model is configured
        
        # Reconfigure Tracker 
        self._reconfigure_tracker(attrs)
        
        # Reconfigure Zones 
        self._reconfigure_zones(attrs)    

        self.logger.info("Reconfiguration complete.")

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
        # Classification Logic
        in_queue   = zones[WAIT].contains(point)
        in_fail    = zones[FAILURE].contains(point)
        in_success = zones[SUCCESS].contains(point)
        entered    = zones[ENTER].contains(point) # removed for now

        current_state = LABEL_WALK

        # State Classification
        if in_queue: 
            current_state = LABEL_IN
        elif in_fail:
            current_state = LABEL_FAIL
        elif in_success:
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
        
    def get_floor_centroid(  
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
        tracker_config_location = getattr(self, 'tracker_config_location', None)

        if tracker_config_location:
            self.logger.debug(f"Using tracking mode")
            return await self.get_tracks(image, extra=extra, timeout=timeout)

        else:
            self.logger.debug(f"Using detection-only mode")
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
                    device=self.device,           # Run on CPU/GPU/MPS
                    verbose=False                 # Suppress YOLO output
                )[0]  # Get first result from batch (since we're processing 1 image)
                
            # Graceful handling of no detections
            if results is None or results.boxes is None or len(results.boxes) == 0:
                self.logger.debug("No objects detected in frame")
                return []
            
            num_detections = len(results.boxes)
            self.logger.debug(f"Found {num_detections} detections")
            
            # Process detection results                
            for i, box in enumerate(results.boxes):
                class_id = int(box.cls.item())
                confidence = round(box.conf.item(), 4)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                detection = {
                    "class_name": f"{self.class_names[class_id]}_{i}",  
                    "confidence": confidence,
                    "x_min": x1,
                    "y_min": y1,
                    "x_max": x2,
                    "y_max": y2,
                }
                
                try:
                    detections.append(Detection(**detection))
                except TypeError as e:
                    self.logger.warning(f"Error creating Detection: {str(e)} with data: {detection}")
                    
        except Exception as e:
            self.logger.error(f"Error in detection: {str(e)}")
        
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
        
        if not self._check_model_state():
            self.logger.warning("Model state invalid, falling back to detection-only")
            return await self.get_detections_only(image, extra=extra, timeout=timeout)
        
        try:
            # Convert image
            pil_image = viam_to_pil_image(image)
            
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                try:
                    results = self.model.track(
                        pil_image, 
                        tracker=self.tracker_config_location, 
                        persist=True, 
                        classes=self.detection_classes, 
                        device=self.device,  # This gets passed to your BOTSort automatically
                        verbose=False  # Suppress YOLO output
                    )[0]
                except Exception as track_error:
                    error_msg = str(track_error)
                    self.logger.error(f"Tracking failed: {error_msg}")
                    
                    # If it's the specific ReID compatibility error, try a workaround
                    if "YOLOv5 model originally trained" in error_msg and "reid_embedder.pt" in error_msg:
                        self.logger.warning("ReID model compatibility issue detected")
                        self.logger.info("Attempting workaround: trying with tracker reload...")
                        
                        try:
                            # Force reload the model (sometimes fixes compatibility issues)
                            self.model = YOLO(self.model_location, verbose=False)
                            
                            # Try tracking again
                            results = self.model.track(
                                pil_image, 
                                tracker=self.tracker_config_location, 
                                persist=True, 
                                classes=self.detection_classes, 
                                device=self.device,
                                verbose=False
                            )[0]
                            self.logger.info("Workaround successful - tracking with ReID working")
                            
                        except Exception as retry_error:
                            self.logger.error(f"Workaround failed: {retry_error}")
                            self.logger.info("Falling back to detection-only mode")
                            return await self.get_detections_only(image, extra=extra, timeout=timeout)
                    else:
                        # For other tracking errors, fall back to detection-only
                        self.logger.info("Falling back to detection-only mode")
                        return await self.get_detections_only(image, extra=extra, timeout=timeout)
            
            # Graceful handling of no results
            if results is None or results.boxes is None or len(results.boxes) == 0:
                self.logger.debug("No objects detected in tracking mode")
                return []
            
            # Check if tracking IDs are available
            if results.boxes.id is None:
                self.logger.debug("No tracking IDs assigned - objects may be moving too fast or confidence too low")
                # Fall back to detection-only mode for this frame
                return await self.get_detections_only(image, extra=extra, timeout=timeout)
            
            num_detections = len(results.boxes)
            self.logger.debug(f"Tracking {num_detections} objects with persistent IDs")
            
            # Reset current tracks for each detection call
            if hasattr(self, 'current_tracks'):
                self.current_tracks = {zone: [] for zone in self.current_tracks.keys()}
            
            # Convert track IDs to list for safer iteration
            track_ids = results.boxes.id.int().cpu().tolist()
            
            for i, (xyxy, conf, track_id) in enumerate(
                zip(results.boxes.xyxy, results.boxes.conf, track_ids)
            ):
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                confidence = round(conf.item(), 4)
                
                # Only apply zone-based classification if zones exist
                if hasattr(self, 'zones') and self.zones:
                    queue_state = self.get_floor_centroid(x1, y1, x2, y2, track_id)
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
                    self.logger.debug(f"Tracked: {queue_state} (conf: {confidence})")
                except TypeError as e:
                    self.logger.warning(f"Error creating Detection object: {e} with data: {detection}")

        except KeyError as ke:
            self.logger.error(f"Tracker config error - missing required key: {ke}")
            self.logger.info("Falling back to detection-only mode")
            return await self.get_detections_only(image, extra=extra, timeout=timeout)
        except Exception as e:
            self.logger.error(f"Tracking failed: {e}")
            self.logger.info("Falling back to detection-only mode")
            return await self.get_detections_only(image, extra=extra, timeout=timeout)

        self.logger.debug(f"Returning {len(detections)} tracked detections")
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
            if not hasattr(self, 'tracker_config_location') or not self.tracker_config_location:            
                self.logger.debug(f"Tracker is not enabled.")
                return {"error": "Tracking not enabled"}
            else:
                self.logger.debug(f"Current state labels: {self.current_tracks}")
                # Return the current state labels
                return {"current_tracks": self.current_tracks} 

        else:
            return {"error": "Command not recognized"}