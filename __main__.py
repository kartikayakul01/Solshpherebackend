"""
Main entry point for the backend_stub package.
Run with: python -m backend_stub
"""
from .app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_stub.app:app", host="0.0.0.0", port=8000, reload=True)
