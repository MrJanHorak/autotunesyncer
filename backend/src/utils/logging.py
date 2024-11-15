import logging
import sys

def setup_logging(level: int = logging.DEBUG) -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('video_processor.log')
        ]
    )

    # Set third-party loggers to a higher level
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)