"""
Smart Meal Planner Web Application

Features:
    - User signup
    - User login
    - User logout
    - Protected meal planner
    - Gemini AI meal-planning agent
    - Per-user conversation history
    - Per-user meal-plan state
    - Gemini quota/rate-limit handling

Run:
    python web_app.py

Then open:
    http://127.0.0.1:5000
"""

import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    redirect,
    request,
    session,
    send_from_directory,
    url_for,
)
from google import genai
from werkzeug.security import check_password_hash, generate_password_hash

import meal_planner_agent as agent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(override=True)

HERE = Path(__file__).resolve().parent

app = Flask(__name__)

# IMPORTANT:
# Change this to your own random secret before submitting the project.
#
# For a college demo this is fine.
# For production, store it in an environment variable.
app.secret_key = "smart-meal-planner-ca2-secret-key-2026"

# Gemini client.
# The Gemini API key stays on the server.
client = genai.Client()


# ---------------------------------------------------------------------------
# In-memory user database
# ---------------------------------------------------------------------------
#
# This is intentionally simple for your CA2 project.
#
# Restarting Flask will clear these accounts.
#
# Example:
# USERS = {
#     "student@example.com": {
#         "name": "Student",
#         "password": "hashed-password"
#     }
# }
# ---------------------------------------------------------------------------

USERS = {}


# ---------------------------------------------------------------------------
# Per-user application state
# ---------------------------------------------------------------------------
#
# Each user gets:
#
#     conversation
#     current_plan
#
# This prevents different users from sharing meal-planning conversations.
# ---------------------------------------------------------------------------

USER_DATA = {}

LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_user_key():
    """
    Return a unique identifier for the currently logged-in user.
    """

    return session.get("user_email")


def get_user_data():
    """
    Get or create the logged-in user's application state.
    """

    email = get_user_key()

    if not email:
        return None

    if email not in USER_DATA:
        USER_DATA[email] = {
            "conversation": [],
            "current_plan": None,
        }

    return USER_DATA[email]


def is_logged_in():
    """
    Check whether the current browser session is authenticated.
    """

    return "user_email" in session


def is_quota_error(error):
    """
    Detect Gemini free-tier quota/rate-limit errors.
    """

    text = str(error).lower()

    return (
        "429" in text
        or "resource_exhausted" in text
        or "quota" in text
        or "rate limit" in text
        or "ratelimit" in text
    )


def quota_message():
    """
    Friendly message for Gemini quota errors.
    """

    return (
        "⚠️ **Gemini quota temporarily reached**\n\n"
        "The free Gemini API limit has temporarily been reached.\n\n"
        "Please wait about **1 minute** and try again.\n\n"
        "Your existing meal plan is still saved in this session."
    )


# ---------------------------------------------------------------------------
# Login / Signup pages
# ---------------------------------------------------------------------------

@app.get("/login")
def login_page():
    """
    Show the login page.
    """

    if is_logged_in():
        return redirect(url_for("home"))

    return send_from_directory(HERE, "login.html")


@app.get("/signup")
def signup_page():
    """
    Show the signup page.
    """

    if is_logged_in():
        return redirect(url_for("home"))

    return send_from_directory(HERE, "signup.html")


# ---------------------------------------------------------------------------
# Signup API
# ---------------------------------------------------------------------------

@app.post("/signup")
def signup():
    """
    Create a new user account.
    """

    data = request.get_json(silent=True) or {}

    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    # Basic validation
    if not name:
        return jsonify(
            success=False,
            error="Please enter your name."
        ), 400

    if not email or "@" not in email:
        return jsonify(
            success=False,
            error="Please enter a valid email address."
        ), 400

    if len(password) < 6:
        return jsonify(
            success=False,
            error="Password must contain at least 6 characters."
        ), 400

    with LOCK:

        if email in USERS:
            return jsonify(
                success=False,
                error="An account with this email already exists."
            ), 409

        # Never store the actual password.
        USERS[email] = {
            "name": name,
            "password": generate_password_hash(password),
        }

        # Create empty application state.
        USER_DATA[email] = {
            "conversation": [],
            "current_plan": None,
        }

    # Automatically log the user in after signup.
    session.clear()

    session["user_email"] = email
    session["user_name"] = name

    return jsonify(
        success=True,
        message="Account created successfully."
    )


# ---------------------------------------------------------------------------
# Login API
# ---------------------------------------------------------------------------

@app.post("/login")
def login():
    """
    Authenticate an existing user.
    """

    data = request.get_json(silent=True) or {}

    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify(
            success=False,
            error="Enter your email and password."
        ), 400

    user = USERS.get(email)

    if not user:
        return jsonify(
            success=False,
            error="Incorrect email or password."
        ), 401

    if not check_password_hash(user["password"], password):
        return jsonify(
            success=False,
            error="Incorrect email or password."
        ), 401

    # Prevent session fixation.
    session.clear()

    session["user_email"] = email
    session["user_name"] = user["name"]

    # Make sure user state exists.
    get_user_data()

    return jsonify(
        success=True,
        message="Login successful."
    )


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------

@app.get("/logout")
def logout():
    """
    Log the current user out.
    """

    session.clear()

    return redirect(url_for("login_page"))


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    """
    Protected Smart Meal Planner page.
    """

    if not is_logged_in():
        return redirect(url_for("login_page"))

    return send_from_directory(HERE, "index.html")


# ---------------------------------------------------------------------------
# Current user information
# ---------------------------------------------------------------------------

@app.get("/me")
def current_user():
    """
    Return information about the currently logged-in user.

    Useful for displaying:
        Welcome, Rahul
    """

    if not is_logged_in():
        return jsonify(
            logged_in=False
        ), 401

    return jsonify(
        logged_in=True,
        name=session.get("user_name"),
        email=session.get("user_email")
    )


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------

@app.post("/chat")
def chat():
    """
    Send a message to the Smart Meal Planner agent.
    """

    if not is_logged_in():
        return jsonify(
            error="Please log in first."
        ), 401

    data = request.get_json(silent=True) or {}

    text = (data.get("message") or "").strip()

    if not text:
        return jsonify(
            error="Type a message first."
        ), 400

    with LOCK:

        user_data = get_user_data()

        if user_data is None:
            return jsonify(
                error="Your session has expired. Please log in again."
            ), 401

        messages = user_data["conversation"]

        # Add user message.
        messages.append({
            "role": "user",
            "content": text
        })

        activities = []

        def record_activity(activity):
            activities.append(activity)

        try:

            # State object shared with the agent.
            state = {
                "current_plan": user_data.get("current_plan")
            }

            reply = agent.run_turn(
                client,
                messages,
                activity_callback=record_activity,
                state=state
            )

            # Keep the structured plan returned by the agent.
            user_data["current_plan"] = state.get("current_plan")

        except Exception as e:

            # Remove failed user message.
            messages.pop()

            # Friendly Gemini quota message.
            if is_quota_error(e):

                return jsonify(
                    error=quota_message(),
                    activities=activities
                ), 429

            print("Planner error:", repr(e))

            return jsonify(
                error=f"The planner hit a problem: {e}",
                activities=activities
            ), 500

    return jsonify(
        reply=reply,
        activities=activities
    )


# ---------------------------------------------------------------------------
# Reset conversation
# ---------------------------------------------------------------------------

@app.post("/reset")
def reset():
    """
    Start a fresh meal-planning conversation for the current user.
    """

    if not is_logged_in():
        return jsonify(
            error="Please log in first."
        ), 401

    with LOCK:

        user_data = get_user_data()

        if user_data is not None:
            user_data["conversation"] = []
            user_data["current_plan"] = None

    return jsonify(
        ok=True
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Simple endpoint to check whether Flask is running.
    """

    return jsonify(
        status="ok",
        app="Smart Meal Planner"
    )


# ---------------------------------------------------------------------------
# Run server
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print()
    print("🍱 Smart Meal Planner")
    print("---------------------")
    print("Login:   http://127.0.0.1:5000/login")
    print("Signup:  http://127.0.0.1:5000/signup")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )