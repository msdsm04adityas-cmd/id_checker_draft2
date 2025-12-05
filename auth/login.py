from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from db.database import SessionLocal
from db import crud

templates = Jinja2Templates(directory="templates")
router = APIRouter()


# ---------------------------------------------
# LOGIN PAGE (GET)
# ---------------------------------------------
@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


# ---------------------------------------------
# LOGIN PROCESS (POST)
# ---------------------------------------------
@router.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    db = SessionLocal()

    # Check with DB
    user = crud.authenticate(db, username, password)
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid Credentials"}
        )

    # Set cookies
    response = RedirectResponse("/dashboard", status_code=302)
    response.set_cookie("auth", "yes")
    response.set_cookie("user_id", str(user.id))
    return response
