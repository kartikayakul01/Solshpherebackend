#!/bin/bash

# Initialize the database
echo "Initializing database..."
python -c "from database import models; from database.database import engine; models.Base.metadata.create_all(bind=engine)"

# Start the application
echo "Starting application..."
python -m uvicorn backend_stub.main:app --host 0.0.0.0 --port 10000
