from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Dict, Any, List, Optional, Literal, Union

class ReportBase(BaseModel):
    machine_id: str = Field(..., description="Unique identifier for the machine")
    hostname: str = Field(..., description="Hostname of the machine")
    os: str = Field(..., description="Operating system name and version")
    checks: Dict[str, Any] = Field(..., description="Health check results in JSON format")

class ReportCreate(ReportBase):
    pass

class Report(ReportBase):
    id: int
    timestamp: datetime
    
    class Config:
        from_attributes = True
        
    @field_validator('checks', mode='before')
    def validate_checks(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Checks must be a dictionary")
        return v

class SimplifiedReport(BaseModel):
    """Simplified report format for listing reports"""
    id: int
    machine_id: str
    hostname: str
    os: str
    timestamp: datetime
    issues: List[str] = Field(..., description="List of check names with issues or warnings")
    status: Literal["OK", "warning", "error"] = Field(..., description="Overall system status")
    
    class Config:
        from_attributes = True

class ReportListResponse(BaseModel):
    items: List[Union[Report, SimplifiedReport]]
    total: int
    page: int
    limit: int
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

class HealthCheck(BaseModel):
    status: str
    database: str
    version: str = "1.0.0"