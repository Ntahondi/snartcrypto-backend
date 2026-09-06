from fastapi import FastAPI, HTTPException, Request  # ← Make sure Request is imported
from pydantic import BaseModel
from snailguard import protect_fastapi

app = FastAPI(title="SnailGuard AI Protected API", version="1.0.0")

class LoginRequest(BaseModel):
    username: str
    password: str

class UserData(BaseModel):
    user_id: str
    sensitive_data: str

@app.get("/")
async def root():
    return {"message": "FastAPI protected by SnailGuard AI"}

@app.get("/api/public")
async def public_endpoint():
    return {"data": "This is a public endpoint"}

@app.get("/api/protected")
@protect_fastapi()
async def protected_endpoint(request: Request):  # ← ADD THIS PARAMETER
    return {
        "data": "This endpoint is protected by SnailGuard AI",
        "features": ["zero_fp", "cascade_detection", "multi_ai_models"]
    }

@app.post("/api/login")
@protect_fastapi(enable_economic_warfare=True)
async def login(credentials: LoginRequest, request: Request):  # ← ADD request PARAMETER
    # This endpoint has economic warfare enabled
    return {
        "status": "login_processed",
        "protection": "economic_warfare_active", 
        "user": credentials.username
    }

@app.post("/api/sensitive-data")
@protect_fastapi()
async def sensitive_data(user_data: UserData, request: Request):  # ← ADD request PARAMETER
    return {
        "status": "data_stored",
        "protection_level": "maximum",
        "user_id": user_data.user_id
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI app with SnailGuard AI...")
    uvicorn.run(app, host="0.0.0.0", port=8000)