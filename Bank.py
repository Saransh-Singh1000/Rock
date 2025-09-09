from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import uvicorn
import sqlite3
import os
from cryptography.fernet import Fernet

app = FastAPI()

# ===== HTML UI =====
html_page = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bank Signup</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, red, pink);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0;
        }
        .card {
            background: rgba(255, 255, 255, 0.7);
            border-radius: 15px;
            padding: 40px;
            width: 350px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        }
        h2 {
            text-align: center;
            margin-bottom: 20px;
            color: black;
        }
        .form-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
        }
        label {
            width: 40%;
            text-align: left;
            font-weight: bold;
            color: black;
        }
        input {
            width: 55%;
            padding: 8px;
            border: none;
            border-radius: 6px;
            outline: none;
        }
        button {
            background-color: #007BFF;
            color: white;
            border: none;
            padding: 12px;
            width: 100%;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s ease-in-out;
        }
        button:hover {
            transform: scale(1.1);
        }
        .error {
            color: red;
            font-weight: bold;
            text-align: center;
            margin-top: 15px;
        }
        .success {
            color: green;
            font-weight: bold;
            text-align: center;
            margin-top: 15px;
        }
    </style>
</head>
<body>
    <div class="card">
        <h2>Sign Up</h2>
        <form action="/signup" method="post">
            <div class="form-row">
                <label for="username">Username:</label>
                <input type="text" id="username" name="username" required>
            </div>
            <div class="form-row">
                <label for="password">Password:</label>
                <input type="password" id="password" name="password" required>
            </div>
            <button type="submit">Sign Up</button>
        </form>
    </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return HTMLResponse(content=html_page)


@app.post("/signup")
def signup(username: str = Form(...), password: str = Form(...)):
    # Root data folder
    root_folder = "DataFolder"
    os.makedirs(root_folder, exist_ok=True)

    # User folder structure
    user_folder = os.path.join(root_folder, username)
    key_folder = os.path.join(user_folder, "Key")
    data_folder = os.path.join(user_folder, "Data")
    key_file = os.path.join(key_folder, "EncryptionKey.key")
    db_file = os.path.join(data_folder, "UserData.db")

    # Check if user already exists
    if os.path.exists(user_folder):
        return HTMLResponse(
            f"""
            <h1 class="error">❌ User already exists, change your name.</h1>
            <br><a href='/'>🔙 Back</a>
            """,
            status_code=400
        )

    # Create folders
    os.makedirs(key_folder, exist_ok=True)
    os.makedirs(data_folder, exist_ok=True)

    # Generate encryption key
    encryption_key = Fernet.generate_key()
    with open(key_file, "wb") as f:
        f.write(encryption_key)
    cipher = Fernet(encryption_key)

    # Encrypt username & password
    enc_username = cipher.encrypt(username.encode()).decode()
    enc_password = cipher.encrypt(password.encode()).decode()

    # Create database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL
        )
    """)
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (enc_username, enc_password))
    conn.commit()
    conn.close()

    print(f"✅ New user signed up: {username}")

    return HTMLResponse(
        f"""
        <h1 class="success">✅ User {username} created successfully!</h1>
        <br><a href='/'>🔙 Back</a>
        """
    )


if __name__ == "__main__":
    uvicorn.run("Bank:app", host="127.0.0.1", port=8000, reload=True)
