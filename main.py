# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio
import uvicorn
import psutil

# Import your original function here
from temp import scrape_linkedin_profiles

app = FastAPI()

class ScrapeRequest(BaseModel):
    search_query: str
    location: Optional[str] = "Nigeria"
    max_profiles: Optional[int] = 5
    job_spec: dict


@app.post("/scrape" )
async def scrape_profiles(request: ScrapeRequest):
    try:
        result = await scrape_linkedin_profiles(
            search_query=request.search_query,
            location=request.location,
            max_profiles=request.max_profiles,
            job_spec = request.job_spec
        )
        
        print(f"CPU Usage: {psutil.cpu_percent(interval=1, percpu=True)}%")
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)

