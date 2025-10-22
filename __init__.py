from .nodes.LoadImagesFromDirectory import LoadImagesFromDirectory
from .nodes.WanKeyframeToVideo import WanKeyframeToVideo

NODE_CLASS_MAPPINGS = {
    "LoadImagesFromDirectory": LoadImagesFromDirectory,
    "WanKeyframeToVideo": WanKeyframeToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImagesFromDirectory": "Load Images From Directory",
    "WanKeyframeToVideo": "Wan Keyframe To Video",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]