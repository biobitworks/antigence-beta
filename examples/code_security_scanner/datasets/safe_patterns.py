"""
Safe Code Patterns Dataset

Curated examples of secure Python code demonstrating best practices.
These patterns represent "self" in the immune system metaphor -
normal, trusted code that the system should recognize as safe.
"""

SAFE_PATTERNS = [
    # ========== DATABASE - Parameterized Queries (10 examples) ==========
    {
        "code": """import sqlite3
conn = sqlite3.connect('database.db')
cursor = conn.cursor()
cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
result = cursor.fetchone()""",
        "label": "safe",
        "category": "database",
        "description": "Parameterized SQL query with placeholder"
    },
    {
        "code": """import psycopg2
conn = psycopg2.connect(dbname='mydb')
cursor = conn.cursor()
cursor.execute('INSERT INTO logs (timestamp, level, message) VALUES (%s, %s, %s)',
               (timestamp, level, message))
conn.commit()""",
        "label": "safe",
        "category": "database",
        "description": "PostgreSQL parameterized insert"
    },
    {
        "code": """from sqlalchemy import text
result = session.execute(
    text('SELECT * FROM users WHERE email = :email'),
    {'email': user_email}
)""",
        "label": "safe",
        "category": "database",
        "description": "SQLAlchemy parameterized query"
    },
    {
        "code": """import mysql.connector
cursor = connection.cursor(prepared=True)
query = 'UPDATE users SET last_login = %s WHERE id = %s'
cursor.execute(query, (datetime.now(), user_id))""",
        "label": "safe",
        "category": "database",
        "description": "MySQL prepared statement"
    },
    {
        "code": """from django.db import connection
with connection.cursor() as cursor:
    cursor.execute('SELECT * FROM app_users WHERE username = %s', [username])
    row = cursor.fetchone()""",
        "label": "safe",
        "category": "database",
        "description": "Django raw SQL with parameters"
    },

    # ========== FILE I/O - Safe Operations (10 examples) ==========
    {
        "code": """from pathlib import Path
config_path = Path('/app/config/settings.json')
with config_path.open('r') as f:
    config = json.load(f)""",
        "label": "safe",
        "category": "file_io",
        "description": "Safe file reading with pathlib"
    },
    {
        "code": """import os
safe_directory = '/data/uploads'
os.makedirs(safe_directory, exist_ok=True)
filepath = os.path.join(safe_directory, 'file.txt')
with open(filepath, 'w') as f:
    f.write(content)""",
        "label": "safe",
        "category": "file_io",
        "description": "Safe file writing in designated directory"
    },
    {
        "code": """import tempfile
with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
    temp_file.write(data)
    temp_path = temp_file.name""",
        "label": "safe",
        "category": "file_io",
        "description": "Using temp files safely"
    },
    {
        "code": """import json
from pathlib import Path

data_dir = Path(__file__).parent / 'data'
json_file = data_dir / 'config.json'
if json_file.exists():
    with json_file.open('r') as f:
        config = json.load(f)""",
        "label": "safe",
        "category": "file_io",
        "description": "Safe path construction and validation"
    },
    {
        "code": """import csv
with open('data.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        process_row(row)""",
        "label": "safe",
        "category": "file_io",
        "description": "Safe CSV reading"
    },

    # ========== INPUT VALIDATION (8 examples) ==========
    {
        "code": """import re
def validate_username(username):
    if not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
        raise ValueError('Invalid username format')
    return username""",
        "label": "safe",
        "category": "validation",
        "description": "Username validation with regex"
    },
    {
        "code": """def validate_email(email):
    from email_validator import validate_email as validator
    try:
        valid = validator(email)
        return valid.email
    except EmailNotValidError:
        raise ValueError('Invalid email address')""",
        "label": "safe",
        "category": "validation",
        "description": "Email validation with library"
    },
    {
        "code": """from pydantic import BaseModel, validator

class UserInput(BaseModel):
    age: int
    email: str

    @validator('age')
    def age_must_be_positive(cls, v):
        if v < 0 or v > 150:
            raise ValueError('Invalid age')
        return v""",
        "label": "safe",
        "category": "validation",
        "description": "Pydantic input validation"
    },
    {
        "code": """def sanitize_input(user_input):
    import html
    return html.escape(user_input)

safe_output = sanitize_input(request.form['comment'])""",
        "label": "safe",
        "category": "validation",
        "description": "HTML escaping for XSS prevention"
    },
    {
        "code": """user_id = request.args.get('id')
try:
    user_id = int(user_id)
    if user_id < 0:
        raise ValueError('ID must be positive')
except (ValueError, TypeError):
    return error_response('Invalid user ID')""",
        "label": "safe",
        "category": "validation",
        "description": "Type validation and error handling"
    },

    # ========== API CALLS & REQUESTS (8 examples) ==========
    {
        "code": """import requests
response = requests.get(
    'https://api.example.com/users',
    params={'query': user_input},
    timeout=10
)
data = response.json()""",
        "label": "safe",
        "category": "api",
        "description": "Safe API call with parameters"
    },
    {
        "code": """import os
import requests

API_KEY = os.getenv('API_KEY')
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}
response = requests.post(url, headers=headers, json=payload)""",
        "label": "safe",
        "category": "api",
        "description": "API authentication from environment"
    },
    {
        "code": """import httpx

async with httpx.AsyncClient() as client:
    response = await client.get(
        'https://api.example.com/data',
        params={'id': item_id}
    )
    return response.json()""",
        "label": "safe",
        "category": "api",
        "description": "Async HTTP client with parameters"
    },
    {
        "code": """from urllib.parse import urlencode
base_url = 'https://api.example.com/search'
params = {'q': search_term, 'limit': 10}
url = f'{base_url}?{urlencode(params)}'
response = requests.get(url)""",
        "label": "safe",
        "category": "api",
        "description": "URL encoding for safe parameters"
    },

    # ========== SUBPROCESS - Safe Execution (6 examples) ==========
    {
        "code": """import subprocess
result = subprocess.run(
    ['ls', '-la', directory],
    capture_output=True,
    check=True,
    text=True
)
print(result.stdout)""",
        "label": "safe",
        "category": "subprocess",
        "description": "Safe subprocess with list arguments"
    },
    {
        "code": """import subprocess
result = subprocess.run(
    ['git', 'log', '--oneline', '-n', '10'],
    cwd='/repo/path',
    capture_output=True,
    timeout=30
)""",
        "label": "safe",
        "category": "subprocess",
        "description": "Safe git command execution"
    },
    {
        "code": """import shlex
import subprocess

# Even if we must use shell, sanitize inputs
safe_filename = shlex.quote(filename)
result = subprocess.run(
    f'cat {safe_filename}',
    shell=True,
    capture_output=True
)""",
        "label": "safe",
        "category": "subprocess",
        "description": "Shell quoting when shell=True necessary"
    },

    # ========== CRYPTOGRAPHY - Best Practices (8 examples) ==========
    {
        "code": """from cryptography.fernet import Fernet
key = Fernet.generate_key()
cipher = Fernet(key)
encrypted = cipher.encrypt(plaintext.encode())""",
        "label": "safe",
        "category": "crypto",
        "description": "Fernet symmetric encryption"
    },
    {
        "code": """import hashlib
import os

def hash_password(password):
    salt = os.urandom(32)
    hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return salt + hash_obj""",
        "label": "safe",
        "category": "crypto",
        "description": "PBKDF2 password hashing with salt"
    },
    {
        "code": """from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    length=32,
    salt=salt,
    iterations=100000,
)
key = kdf.derive(password.encode())""",
        "label": "safe",
        "category": "crypto",
        "description": "Modern KDF usage"
    },
    {
        "code": """import bcrypt
password = b'supersecret'
salt = bcrypt.gensalt()
hashed = bcrypt.hashpw(password, salt)""",
        "label": "safe",
        "category": "crypto",
        "description": "Bcrypt password hashing"
    },
    {
        "code": """from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048
)""",
        "label": "safe",
        "category": "crypto",
        "description": "RSA key generation"
    },

    # ========== CONFIGURATION & SECRETS (5 examples) ==========
    {
        "code": """import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
SECRET_KEY = os.getenv('SECRET_KEY')

if not SECRET_KEY:
    raise ValueError('SECRET_KEY not found in environment')""",
        "label": "safe",
        "category": "config",
        "description": "Loading secrets from environment"
    },
    {
        "code": """from pathlib import Path
import yaml

config_file = Path(__file__).parent / 'config.yaml'
with config_file.open('r') as f:
    config = yaml.safe_load(f)""",
        "label": "safe",
        "category": "config",
        "description": "Safe YAML loading"
    },
    {
        "code": """from configparser import ConfigParser

config = ConfigParser()
config.read('/etc/app/config.ini')
database_host = config.get('database', 'host')""",
        "label": "safe",
        "category": "config",
        "description": "INI config file parsing"
    },
]

# Add metadata
for i, pattern in enumerate(SAFE_PATTERNS):
    pattern['id'] = f'safe_{i + 1:03d}'
    if 'tags' not in pattern:
        pattern['tags'] = []
