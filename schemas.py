# ====== ISSUE IDENTIFIED AND FIXED ======

# PROBLEM: The current code might not return all parsed CV data due to:
# 1. Schema conversion issues between dict and ProfileData model
# 2. Missing fields in the response model
# 3. Data loss during processing

# ====== app/models/schemas.py (COMPLETE UPDATED VERSION) ======
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class ContactInfo(BaseModel):
    email: Optional[str] = ""
    phone: Optional[str] = ""
    linkedin: Optional[str] = ""
    website: Optional[str] = ""
    portfolio: Optional[str] = ""

class ProfileData(BaseModel):
    # ENSURE ALL FIELDS FROM ULTRA-ROBUST PARSER ARE INCLUDED
    name: str
    location: Optional[str] = ""
    contact_information: ContactInfo
    top_skills: List[str] = []
    certifications: List[str] = []
    honors_awards: List[str] = []
    summary: Optional[str] = ""
    experience: List[str] = []
    education: List[str] = []
    languages: List[str] = []  # This was added in ultra-robust parser
    
    # Processing metadata
    source_pdf: Optional[str] = None
    download_timestamp: Optional[str] = None
    
    # Add model config to handle extra fields and case conversion
    class Config:
        # Allow extra fields that might come from the parser
        extra = "allow"
        # Handle field name variations
        allow_population_by_field_name = True


