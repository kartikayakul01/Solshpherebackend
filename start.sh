#!/bin/bash
# Initialize the database
echo "Initializing database..."
python -c "from database import models; from database.database import engine; models.Base.metadata.create_all(bind=engine)"

# Start the application
echo "Starting application..."
uvicorn backend_stub.app:app --host 0.0.0.0 --port $PORT
