"""Backend stub package for testing the System Health Agent."""
__version__ = "0.1.0"

# Import the FastAPI app to make it available when importing the package
from .app import app

__all__ = ["app"]
