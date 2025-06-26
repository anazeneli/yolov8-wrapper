import os, sys

from pathlib import Path
from typing import ClassVar, Final, List, Mapping, Optional, Sequence, Tuple, cast
from typing_extensions import Self
from urllib.request import urlretrieve

from viam.media.video import ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import (Classification, Detection,
                                       GetPropertiesResponse)
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import *
from viam.utils import ValueTypes

from viam.utils import struct_to_dict
from ultralytics.engine.results import Results
from viam.components.camera import Camera, ViamImage
from viam.media.utils.pil import viam_to_pil_image
from shapely.geometry import Point, Polygon


from ultralytics import YOLO
import torch

from viam.logging import getLogger

LOGGER = getLogger(__name__)

MODEL_DIR = os.environ.get(
    "VIAM_MODULE_DATA", os.path.join(os.path.expanduser("~"), ".data", "models")
) 

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

class Yolov8Tracker(Vision, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(ModelFamily("azeneli", "yolov8-tracker"), "yolov8-tracker")

    def __init__(self, name: str):
        super().__init__(name=name)
        self.camera_name = None
        self.state_labels: dict[int, str] = {} 
        # Return current people in tracks 
        self.current_tracks = {}  # Track ID to state mapping

        
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

        # Validate detector model location 
        if "model_location" not in attrs or not isinstance(attrs["model_location"], str):
            raise ValueError("model_location is a required path to your model weights file (.pt) ")
        
        # # Optional dependencies 
        if "tracker_config_location" in attrs:
            LOGGER.info(f"Tracking enabled")
            
            if not isinstance(attrs["tracker_config_location"], str):
                raise ValueError("tracker_config_location must be a string path to your config file")

        # Validate user-defined field (zones)
        if "zones" not in attrs or not isinstance(attrs["zones"], dict):
            raise ValueError("zones is required and must be a dictionary of polygon list coordinates")
        

        return required_dependencies, []


    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        # self.dependencies = dependencies

        attrs = struct_to_dict(config.attributes) 

        # Camera 
        camera_component = dependencies.get(Camera.get_resource_name(str(attrs.get("camera_name"))))
        #self.camera = cast(Camera, camera_component)
        self.camera = camera_component
        # Grab path to video folder of images 
        # Load torch codec to load in images + inference at start up 
        # For demo, run inference ahead of time? 

        model_location = str(attrs.get("model_location"))

        self.task = str(attrs.get("task")) or None

        self.enable_tracker = False

        if self.is_path(model_location):
            self.model = YOLO(model_location, task=self.task)

        try: 
            tracker_config_location = str(attrs.get("tracker_config_location", ""))
            if not tracker_config_location:
                raise Exception("tracker_config_location is required when tracker is enabled")

            self.TRACKER_PATH = os.path.abspath(
                    tracker_config_location 
            )
            self.check_path(self.TRACKER_PATH)
            self.logger.info("Tracker enabled and initialized")

            self.enable_tracker = True

        except Exception as e:
            raise Exception(f"Tracker configuration failed: {str(e)}") 
        
        if self.enable_tracker: 
            # Check for zones 
            if "zones" in attrs: 
                # Prepare zones for tracking use 
                self.zones = self.prepare_zones(attrs["zones"])
                self.logger.debug(f"Zones prepared: {self.zones}")


            
        # Initialize current tracks
        if self.zones: 
            # Initialize zones and current tracks
            self.current_tracks = {zone: [] for zone in self.zones.keys()}
            # Initialize state for walking by 
            self.current_tracks[WALK] = []  # Track ID to state mapping for walking by
            LOGGER.info(f"CURRENT TRACKS  {self.current_tracks}") 

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


    def get_centroid(self, points: list) -> tuple[int, int]:
        """Calculate the centroid of a list of points."""
        if not points:
            return None
        cx = int(sum(p[0] for p in points) / len(points))
        cy = int(sum(p[1] for p in points) / len(points))
        
        return cx, cy


    def prepare_zones(self, zones):
        """
            Prepare and process zone data.

            Args:
                zones (dict): A dictionary where keys are zone names (str) and values are numpy arrays representing polygons.

            Returns:
                dict: Processed zones with the same structure.
        """ 
        # convert to numpy arrays
        LOGGER.info(f"ZONES {zones}")
        for zone_name, polygon in zones.items():            
            zones[zone_name] = Polygon(polygon)

        return zones


    def classify_by_feet_centroid(self, point: tuple[Point, Point], prev_label: int, zones: dict) -> str:
        # ── Classification Logic ──
        in_queue   = zones[WAIT].contains(point)
        in_fail    = zones[FAILURE].contains(point)
        in_success = zones[SUCCESS].contains(point)
        entered    = zones[ENTER].contains(point) # removed for now
        # walking_by = zones[WALK].contains(point) # unnecessary, handled by default state
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


        # current_state = prev_label
        
        # if in_queue:
        #     # Only set in queue if we saw them "ENTER" through queue entrance 
        #     # OR if "in_queue for n frames"
        #     if not prev_label == LABEL_IN: # : and entered: 
        #     #     current_state = LABEL_IN
        #     # elif prev_label == LABEL_IN and entered: # likely an exit 
        #         current_state = LABEL_IN
        #     else: 
        #         # Likely still noise
        #         current_state = prev_label
        # elif in_fail:
        #     if prev_label == LABEL_IN or prev_label == LABEL_FAIL:
        #         current_state = LABEL_FAIL
        #     else: 
        #         current_state = prev_label

        # elif in_success: 
        #     current_state = LABEL_SUCCESS
        # else: 
        #     current_state = LABEL_WALK

        return current_state


    def is_path(self, path: str) -> bool:
        try:
            Path(path)
            return os.path.exists(path)
        except ValueError:
            return False

    def check_path(self, path):
        """ 
            Check if path exists on Viam machine
        """ 
        if not os.path.exists(path):
            self.logger.debug(f"Tracker path expected in model path folder.{path}")
            raise FileExistsError(path)
             

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
        image = await self.camera.get_image(mime_type="image/jpeg")

        return await self.get_detections(image)
    

    def get_current_state(self, track_id, keypoints): 
        foot_pts = [point for point in keypoints[13:17] if point[0] > 5 and point[1] > 5] # feet & knees
        current_state = LABEL_WALK
        prev_label = self.state_labels.get(track_id, LABEL_WALK)

        if len(foot_pts) < 2:
            return prev_label

        cx, cy = self.get_centroid(foot_pts)
        current_shapely_point = Point((cx, cy))
        
        current_state = self.classify_by_feet_centroid(current_shapely_point, prev_label, self.zones)
        self.state_labels[track_id] = current_state 

        return current_state


    async def get_detections(self, image: ViamImage, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None) -> List[Detection]:
        detections = [] 

        try:
            # Convert to pil image for tracker
            pil_image = viam_to_pil_image(image)  # Convert ViamImage to PIL image

            results = self.model.track(pil_image, tracker=self.TRACKER_PATH, persist=True, classes=[0], device=self.device)[0]
            LOGGER.info(f"results {results}")

            if results is None or len(results.boxes) == 0:
                self.logger.debug("No results or bounding boxes found.")
                return detections

            # Reset current tracks for each detection call 
            self.current_tracks = {zone: [] for zone in self.current_tracks.keys()}  
            
            # Total detections
            self.logger.debug(f"Total detections: {len(results.boxes)}")

            for i, (xyxy, conf, track_id) in enumerate(zip(results.boxes.xyxy, results.boxes.conf, results.boxes.id)):
                # self.logger.info(f"Processing detection {i}: xyxy={xyxy}, conf={conf}, track_id={track_id}")

                # Convert bounding box coordinates to integers (cast to int)
                x1, y1, x2, y2 = map(int, xyxy.tolist())  # This will ensure integer values
                confidence = round(conf.item(), 4)
                track_id = int(track_id.item())  # Ensure track_id is an integer

                # # Log bounding box and confidence types
                # self.logger.info(f"Bounding Box Type: {type(x1)}, {type(y1)}, {type(x2)}, {type(y2)}")
                # self.logger.info(f"Confidence Type: {type(confidence)}, Track ID Type: {type(track_id)}")

                if results.keypoints is not None:
                    kpts = results.keypoints.xy[i].tolist()
                    # Get zone state 
                    # TODO: Implement a state change 
                    state = self.get_current_state(track_id, kpts)

                    # Update current tracks with the new state
                    self.logger.debug(f"Track ID {track_id} in state {state} ")
                    self.current_tracks[STATES[state]].append(int(track_id))

                    queue_state = f"{str(track_id)}_{str(state)}"
                    
                else:
                    self.logger.debug(f"No keypoints found for track {track_id}.")
                    continue  # Skip this detection if no keypoints

                
                # Prepare detection data
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
            self.logger.debug(f"Error in get_detections: {str(e)}")

        return detections


    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Classification]:
        pass

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Classification]:
        self.logger.debug("`get_classifications` is not implemented")
        pass

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[PointCloudObject]:
        self.logger.debug("`get_object_point_clouds` is not implemented")
        pass

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> Vision.Properties:

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

        # Creates object for tracking values at the current timestamp
        if command.get("command") == "get_current_tracks":
            # Log the current state labels
            self.logger.info(f"Current state labels: {self.current_tracks}")

            # Return the current state labels
            return {"current_tracks": self.current_tracks} 
        else:
            return {"error": "Command not recognized"}
        

