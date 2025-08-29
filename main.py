"""
Main entry point for the SolSphere backend.
Run with: python -m backend_stub.main
"""
import uvicorn
from fastapi import FastAPI
from app import app

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
