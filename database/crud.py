from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime
from .models import Report

def create_report(db: Session, report_data: Dict[str, Any]) -> Report:
    """Create a new report in the database."""
    db_report = Report(**report_data)
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return db_report

def get_reports(
    db: Session, 
    skip: int = 0, 
    limit: int = 100,
    machine_id: Optional[str] = None,
    status: Optional[str] = None,
    sort = None
) -> List[Report]:
    """Retrieve reports with filtering and sorting.
    
    Args:
        db: Database session
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return
        machine_id: Filter by machine ID
        status: Filter by status (checks.status)
        sort: SQLAlchemy sort expression (e.g., Report.timestamp.desc())
    """
    query = db.query(Report)
    
    # Apply filters
    if machine_id:
        query = query.filter(Report.machine_id == machine_id)
    
    if status:
        # Assuming checks is a JSON/JSONB column with a status field
        query = query.filter(Report.checks['status'].astext == status)
    
    # Apply sorting (default to timestamp desc if not specified)
    if sort is not None:
        query = query.order_by(sort)
    else:
        query = query.order_by(Report.timestamp.desc())
    
    # Apply pagination
    return query.offset(skip).limit(limit).all()

def get_latest_report_by_machine_id(db: Session, machine_id: str) -> Optional[Report]:
    """Get the most recent report for a specific machine.
    
    Args:
        db: Database session
        machine_id: The machine ID to get the latest report for
        
    Returns:
        The most recent Report for the machine, or None if no reports exist
    """
    return (
        db.query(Report)
        .filter(Report.machine_id == machine_id)
        .order_by(Report.timestamp.desc())
        .first()
    )

def get_report_count(
    db: Session, 
    machine_id: Optional[str] = None,
    status: Optional[str] = None
) -> int:
    """Get the total count of reports with optional filtering.
    
    Args:
        db: Database session
        machine_id: Filter by machine ID
        status: Filter by status (checks.status)
    """
    query = db.query(Report)
    
    if machine_id:
        query = query.filter(Report.machine_id == machine_id)
        
    if status:
        query = query.filter(Report.checks['status'].astext == status)
        
    return query.count()

def delete_reports_older_than(db: Session, days: int = 30) -> int:
    """Delete reports older than the specified number of days."""
    from datetime import datetime, timedelta
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    result = db.query(Report).filter(Report.timestamp < cutoff_date).delete()
    db.commit()
    return result
