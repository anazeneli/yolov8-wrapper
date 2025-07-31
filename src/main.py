import asyncio
from viam.module.module import Module
try:
    from src.models.yolov8 import Yolov8
except ModuleNotFoundError:
    # when running as local module with run.sh
    from .models.yolov8 import Yolov8


if __name__ == '__main__':
    asyncio.run(Module.run_from_registry())
