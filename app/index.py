import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader

load_dotenv(override=True)
openai = OpenAI()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# Load LinkedIn PDF
# -------------------
linkedin = ""
try:
    reader = PdfReader("me/linkdin.pdf")
    for page in reader.pages:
        text = page.extract_text()
        if text:
            linkedin += text
except Exception as e:
    print(f"Warning: could not read LinkedIn PDF: {e}")
    linkedin = "LinkedIn profile not available"

github = "https://github.com/pranavhole"
summary = linkedin
name = "Pranav Hole"

system_prompt = f"""You are acting as {name}. You are answering questions on {name}'s website...

## Summary:
{summary}

## LinkedIn Profile:
{linkedin}

## GitHub Profile:
{github}

Stay in character as {name}.
"""

# -------------------
# Pushover utilities
# -------------------
pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

def push(message):
    if pushover_user and pushover_token:
        payload = {"user": pushover_user, "token": pushover_token, "message": message}
        try:
            requests.post(pushover_url, data=payload)
        except Exception as e:
            print(f"Pushover error: {e}")

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording interest from {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question} asked that I couldn't answer")
    return {"recorded": "ok"}


# -------------------
# Tool Schemas
# -------------------
record_user_details_json = {
    "type": "function",
    "function": {
        "name": "record_user_details",
        "description": "Record a user's details for follow-up",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string"},
                "name": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["email"]
        }
    }
}

record_unknown_question_json = {
    "type": "function",
    "function": {
        "name": "record_unknown_question",
        "description": "Record a question the bot could not answer",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"}
            },
            "required": ["question"]
        }
    }
}

tools = [record_user_details_json, record_unknown_question_json]


def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {}
        results.append({
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": tool_call.id
        })
    return results


# -------------------
# HTML Chat UI
# -------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!doctype html>
    <html>
    <head><title>Chat with Pranav Hole</title></head>
    <body>
    <h2>Chatbot</h2>
    <div id="chatbox" style="border:1px solid #ccc; padding:10px; height:300px; overflow-y:scroll;"></div>
    <input id="user_input" autocomplete="off" placeholder="Type a message" style="width: 80%;"/>
    <button onclick="sendMessage()">Send</button>
    <script>
    async function sendMessage() {
        const inputBox = document.getElementById("user_input");
        const chatbox = document.getElementById("chatbox");
        const message = inputBox.value;
        if (!message) return;

        chatbox.innerHTML += '<div><b>You:</b> ' + message + '</div>';
        inputBox.value = "";

        const response = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: message, history: []})
        });

        const reader = response.body.getReader();
        let botReply = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            botReply += new TextDecoder().decode(value);
        }

        chatbox.innerHTML += '<div><b>Bot:</b> ' + botReply + '</div>';
        chatbox.scrollTop = chatbox.scrollHeight;
    }
    </script>
    </body>
    </html>
    """


# -------------------
# Chat Endpoint (Streaming)
# -------------------
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message")
    history = data.get("history", [])

    messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": message}
    ]

    async def generate():
        done = False
        while not done:

            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools
            )

            finish_reason = response.choices[0].finish_reason

            if finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = handle_tool_calls(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
                reply = response.choices[0].message.content
                yield reply

    return StreamingResponse(generate(), media_type="text/plain")


# Run with:  uvicorn main:app --reload
