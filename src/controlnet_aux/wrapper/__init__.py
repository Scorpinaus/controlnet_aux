from .anyline_wrapper import AnylineWrapper
from .canny_wrapper import CannyDetectorWrapper
from .hed_wrapper import HEDWrapper
from .lineart_wrapper import LineartDetectorWrapper
from .lineart_anime_wrapper import LineartAnimeWrapper
from .lineart_standard_wrapper import LineartStandardWrapper
from .zoe_wrapper import ZoeDetectorWrapper
from .TEED_wrapper import TEEDWrapper
from .tiling_wrapper import TilingDetectorWrapper
from .shuffle_wrapper import ContentShuffleWrapper
from .pidinet_wrapper import PidiNetDetectorWrapper
from .normalbae_wrapper import NormalBaeWrapper
from .mlsd_wrapper import MLSDWrapper
from .midas_wrapper import MiDASWrapper
from .leres_wrapper import LeresDetectorWrapper
from .samDetector_wrapper import SamDetectorWrapper
# Import other wrappers here
# from .hed_wrapper import HEDWrapper
# Add any new wrapper imports here

__all__ = [
    'AnylineWrapper',
    'CannyDetectorWrapper',
    'HEDWrapper',
    'LineartDetectorWrapper',
    'LineartAnimeWrapper',
    'LineartStandardWrapper',
    'ZoeDetectorWrapper',
    'TEEDWrapper',
    'TilingDetectorWrapper',
    'ContentShuffleWrapper',
    'PidiNetDetectorWrapper',
    'NormalBaeWrapper',
    'MLSDWrapper',
    'MiDASWrapper',
    'LeresDetectorWrapper',
    'SamDetectorWrapper'
    ]
   # Add other wrapper classes to this list
   # 'HEDWrapper',