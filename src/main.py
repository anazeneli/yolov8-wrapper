import asyncio
from viam.module.module import Module
try:
    from yolov8.src.models.yolov8_tracker import Yolov8Tracker
except ModuleNotFoundError:
    # when running as local module with run.sh
    from .models.yolov8_tracker import Yolov8Tracker


if __name__ == '__main__':
    asyncio.run(Module.run_from_registry())
