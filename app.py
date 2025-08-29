from fastapi import FastAPI, Depends, HTTPException, status, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import desc as sa_desc, asc as sa_asc, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
import os
from dotenv import load_dotenv

# Get the directory where this script is located
script_dir = Path(__file__).parent
env_path = script_dir / '.env'
print(f"Loading .env from: {env_path}")
load_dotenv(env_path, override=True)

# Verify AUTH_TOKEN is loaded
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
if not AUTH_TOKEN:
    raise ValueError("AUTH_TOKEN is not set in environment variables. Please check your .env file.")
print(f"AUTH_TOKEN loaded: {AUTH_TOKEN[:3]}...")  # Print first 3 chars for verification

# Import database models and schemas
try:
    from .database import models
    from .database.schemas import ReportCreate, Report, ReportListResponse, SimplifiedReport
except ImportError:
    # For direct execution
    from database import models
    from database.schemas import ReportCreate, Report, ReportListResponse, SimplifiedReport

# Database connection setup
try:
    from .database.database import engine, SessionLocal
except ImportError:
    # For direct execution
    from database.database import engine, SessionLocal

# Initialize FastAPI app
app = FastAPI(
    title="SolSphere API",
    description="API for managing system health reports",
    version="1.0.0",
)

# Authentication middleware
class BearerAuth(HTTPBearer):
    async def __call__(self, request: Request):
        # Skip auth for health check endpoint
        if request.url.path == "/health":
            return None
            
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        
        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid authentication scheme"
                )
            
            if credentials.credentials != AUTH_TOKEN:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token"
                )
            return credentials.credentials
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )

# Add auth middleware
auth = BearerAuth()

# Add middleware to verify token for all routes
@app.middleware("http")
async def verify_token_middleware(request: Request, call_next: Callable):
    
    # Skip auth for health check endpoint
    if request.url.path != "/health":
        try:
            
            await auth(request)
        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail}
            )
    return await call_next(request)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./reports.db")

# Create database directory if it doesn't exist
db_path = Path("reports.db")
db_path.parent.mkdir(parents=True, exist_ok=True)

# Database session
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create tables
models.Base.metadata.create_all(bind=engine)

# Utility functions
def get_sort_expression(model, sort_by: str, sort_order: str = 'desc'):
    """Get SQLAlchemy sort expression."""
    sort_column = getattr(model, sort_by, None)
    if sort_column is None:
        sort_column = model.timestamp
    return sa_desc(sort_column) if sort_order.lower() == 'desc' else sa_asc(sort_column)

# API Endpoints
def _process_report_checks(checks: Dict[str, Any]) -> tuple[List[str], str]:
    """Process report checks to extract issues and determine overall status.
    
    Returns:
        tuple: (list_of_issues, overall_status)
    """
    if not isinstance(checks, dict):
        return [], "OK"
        
    issues = []
    has_error = False
    has_warning = False
    
    for check_name, check_data in checks.items():
        
        if check_name == 'overall_status':
            continue
            
        if not isinstance(check_data, dict):
            continue
            
        check_status = check_data.get('status', '').lower()
        if check_status == 'issue':
            has_error = True
            issues.append(check_name)
        elif check_status == 'unknown':
            has_warning = True
            issues.append(check_name)
    
    # Determine overall status
    if has_error:
        status = "error"
    elif has_warning:
        status = "warning"
    else:
        status = "OK"
    
    return issues, status

@app.get("/reports/{report_id}", response_model=Report, status_code=status.HTTP_200_OK)
async def get_report(
    report_id: int,
    simplified: bool = Query(False, description="Return simplified report format"),
    db: Session = Depends(get_db)
):
    """Get a single report by ID.
    
    Args:
        report_id: The ID of the report to retrieve.
        simplified: If True, returns a simplified version of the report.
        
    Returns:
        The report with the specified ID in either full or simplified format.
        
    Raises:
        HTTPException: If the report is not found.
    """
    
    db_report = db.query(models.Report).filter(models.Report.id == report_id).first()
    if not db_report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report with ID {report_id} not found"
        )
    
    if simplified:
        issues, status = _process_report_checks(db_report.checks)
        
        return {
            "id": db_report.id,
            "machine_id": db_report.machine_id,
            "hostname": db_report.hostname,
            "os": db_report.os,
            "timestamp": db_report.timestamp,
            "issues": issues,
            "status": status
        }
    
    # Return full report
    return db_report

@app.get("/reports", response_model=ReportListResponse, status_code=status.HTTP_200_OK)
async def list_reports(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    hostname: Optional[str] = Query(None, description="Filter by machine ID or hostname"),
    status: Optional[str] = Query(None, description="Filter by status (OK, warning, error)"),
    sort_by: str = Query("timestamp", description="Field to sort by"),
    sort_order: str = Query("desc", description="Sort order (asc or desc)"),
    detailed: bool = Query(False, description="Return full report details"),
    db: Session = Depends(get_db)
):
    """Get a paginated list of reports with optional filtering and sorting."""
    
    #print all query parameters
    
    print(page,limit,hostname,status,sort_by,sort_order,detailed)
    try:
        offset = (page - 1) * limit
        
        # Build base query
        query = db.query(models.Report)
        
        # Apply filters
        if hostname:
            query = query.filter(
                (models.Report.machine_id == hostname) | 
                (models.Report.hostname.ilike(f'%{hostname}%')) |
                (models.Report.os==hostname)
            )
        
        # Get all repots first to process checks
        all_reports = query.all()
        
        # Process reports to get simplified format
        simplified_reports = []
        
        for report in all_reports:
            
            issues, report_status = _process_report_checks(report.checks)
            
            # Apply status filter if provided
            if status and report_status != status.lower():
                continue
                
            simplified_reports.append({
                "id": report.id,
                "machine_id": report.machine_id,
                "hostname": report.hostname,
                "os": report.os,
                "timestamp": report.timestamp,
                "issues": issues,
                "status": report_status
            })
        # Apply sorting to simplified reports
        reverse_sort = sort_order.lower() == 'desc'
        sort_key = sort_by if sort_by != 'status' else 'status'
        
        # Sort the reports
        try:
            simplified_reports.sort(
                key=lambda x: x.get(sort_key, ''),
                reverse=reverse_sort
            )
        except (TypeError, KeyError):
            # Fallback to timestamp if sort key is invalid
            simplified_reports.sort(
                key=lambda x: x.get('timestamp', ''),
                reverse=reverse_sort
            )
        
        # Apply pagination to simplified reports
        total_items = len(simplified_reports)
        total_pages = (total_items + limit - 1) // limit
        paginated_reports = simplified_reports[offset:offset + limit]
        
        response = {
            "items": paginated_reports,
            "total": total_items,
            "page": page,
            "limit": limit,  # Changed from page_size to limit to match the model
            "total_pages": total_pages
        }
        
        if detailed:
            # If detailed view is requested, fetch full reports for the paginated items
            report_ids = [r['id'] for r in paginated_reports]
            detailed_reports = db.query(models.Report).filter(models.Report.id.in_(report_ids)).all()
            
            # Convert to dict and maintain order
            report_dict = {r.id: r for r in detailed_reports}
            response["items"] = [report_dict[r_id] for r_id in report_ids if r_id in report_dict]
        
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching reports: {str(e)}"
        )

@app.get("/reports/latest", response_model=Report, status_code=status.HTTP_200_OK)
async def get_latest_report(
    machine_id: str = Query(..., description="Machine ID to get the latest report for"),
    db: Session = Depends(get_db)
):
    """Get the latest report for a specific machine."""
    try:
        report = (
            db.query(models.Report)
            .filter(models.Report.machine_id == machine_id)
            .order_by(models.Report.timestamp.desc())
            .first()
        )
        
        if not report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No reports found for machine_id: {machine_id}"
            )
        
        return {
            "id": report.id,
            "machine_id": report.machine_id,
            "hostname": report.hostname,
            "os": report.os,
            "timestamp": report.timestamp.isoformat() if report.timestamp else None,
            "checks": report.checks
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching latest report: {str(e)}"
        )

@app.get("/machines/latest-reports", response_model=List[Report], status_code=status.HTTP_200_OK)
async def get_all_machines_latest_reports(
    db: Session = Depends(get_db)
):
    """Get the latest report for each machine."""
    
    try:
        # Get distinct machine IDs
        machine_ids = db.query(models.Report.machine_id).distinct().all()
        
        latest_reports = []
        for (machine_id,) in machine_ids:
            report = (
                db.query(models.Report)
                .filter(models.Report.machine_id == machine_id)
                .order_by(models.Report.timestamp.desc())
                .first()
            )
            if report:
                latest_reports.append({
                    "id": report.id,
                    "machine_id": report.machine_id,
                    "hostname": report.hostname,
                    "os": report.os,
                    "timestamp": report.timestamp.isoformat() if report.timestamp else None,
                    "checks": report.checks
                })
        
        return latest_reports
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching latest reports: {str(e)}"
        )

@app.get("/machines/{machine_id}/latest", response_model=Report, status_code=status.HTTP_200_OK)
async def get_latest_machine_report(
    machine_id: str,
    db: Session = Depends(get_db)
):
    """Get the latest report for a specific machine by ID."""
    try:
        report = (
            db.query(models.Report)
            .filter(models.Report.machine_id == machine_id)
            .order_by(models.Report.timestamp.desc())
            .first()
        )
        
        if not report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No reports found for machine_id: {machine_id}"
            )
        
        return {
            "id": report.id,
            "machine_id": report.machine_id,
            "hostname": report.hostname,
            "os": report.os,
            "timestamp": report.timestamp.isoformat() if report.timestamp else None,
            "checks": report.checks
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching machine report: {str(e)}"
        )

@app.post("/report", response_model=Report, status_code=status.HTTP_201_CREATED)
async def create_new_report(
    report: ReportCreate,
    db: Session = Depends(get_db)
):
    """Create a new report."""
    try:
        db_report = models.Report(
            machine_id=report.machine_id,
            hostname=report.hostname,
            os=report.os,
            checks=report.checks,
            timestamp=datetime.utcnow()
        )
        
        db.add(db_report)
        db.commit()
        db.refresh(db_report)
        
        return {
            "id": db_report.id,
            "machine_id": db_report.machine_id,
            "hostname": db_report.hostname,
            "os": db_report.os,
            "timestamp": db_report.timestamp.isoformat() if db_report.timestamp else None,
            "checks": db_report.checks
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating report: {str(e)}"
        )

@app.delete("/reports/cleanup", status_code=status.HTTP_200_OK)
async def cleanup_old_reports(
    days: int = Query(30, ge=1, description="Delete reports older than this many days"),
    db: Session = Depends(get_db)
):
    """Delete reports older than the specified number of days."""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        deleted_count = (
            db.query(models.Report)
            .filter(models.Report.timestamp < cutoff_date)
            .delete()
        )
        db.commit()
        
        return {
            "status": "success",
            "message": f"Successfully deleted {deleted_count} reports older than {days} days",
            "deleted_count": deleted_count,
            "cutoff_date": cutoff_date.isoformat()
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cleaning up old reports: {str(e)}"
        )

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint.
    
    Returns:
        dict: Status of the API and its dependencies
    """
    try:
        # Test database connection
        start_time = time.time()
        db.execute("SELECT 1")
        db_latency = (time.time() - start_time) * 1000  # Convert to milliseconds
        db_status = {
            "status": "connected",
            "latency_ms": round(db_latency, 2)
        }
    except Exception as e:
        db_status = {
            "status": "error",
            "message": str(e)
        }
    
    current_time = datetime.utcnow()
    
    return {
        "status": "ok" if db_status["status"] == "connected" else "error",
        "version": "1.0.0",
        "timestamp": current_time.isoformat(),
        "utc_offset": current_time.strftime("%z"),
        "dependencies": {
            "database": db_status
        },
        "metadata": {
            "service": "solsphere-backend",
            "environment": "development"
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)