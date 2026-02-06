"""
Vulnerable Code Patterns Dataset

Curated examples of insecure Python code with known vulnerabilities.
These patterns represent "non-self" threats that the system should detect.
Each example includes CWE classification and severity rating.
"""

VULNERABLE_PATTERNS = [
    # ========== SQL INJECTION (CWE-89) - 6 examples ==========
    {
        "code": """user_id = request.args.get('id')
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)
result = cursor.fetchone()""",
        "label": "vulnerable",
        "vulnerability_type": "sql_injection",
        "severity": "high",
        "cwe_id": "CWE-89",
        "description": "SQL injection via f-string formatting"
    },
    {
        "code": """username = input('Enter username: ')
cursor.execute('SELECT * FROM users WHERE name = \"' + username + '\"')""",
        "label": "vulnerable",
        "vulnerability_type": "sql_injection",
        "severity": "high",
        "cwe_id": "CWE-89",
        "description": "SQL injection via string concatenation"
    },
    {
        "code": """search_term = request.form['search']
query = "SELECT * FROM products WHERE name LIKE '%" + search_term + "%'"
results = db.execute(query)""",
        "label": "vulnerable",
        "vulnerability_type": "sql_injection",
        "severity": "high",
        "cwe_id": "CWE-89",
        "description": "SQL injection in LIKE clause"
    },
    {
        "code": """order_by = request.args.get('sort', 'id')
query = f"SELECT * FROM users ORDER BY {order_by}"
cursor.execute(query)""",
        "label": "vulnerable",
        "vulnerability_type": "sql_injection",
        "severity": "medium",
        "cwe_id": "CWE-89",
        "description": "SQL injection in ORDER BY clause"
    },
    {
        "code": """table_name = get_user_input()
query = "SELECT * FROM " + table_name
cursor.execute(query)""",
        "label": "vulnerable",
        "vulnerability_type": "sql_injection",
        "severity": "critical",
        "cwe_id": "CWE-89",
        "description": "SQL injection with dynamic table name"
    },
    {
        "code": """# Unsafe even with % formatting
user_data = request.form['data']
cursor.execute("INSERT INTO logs VALUES ('%s')" % user_data)""",
        "label": "vulnerable",
        "vulnerability_type": "sql_injection",
        "severity": "high",
        "cwe_id": "CWE-89",
        "description": "SQL injection via % string formatting"
    },

    # ========== COMMAND INJECTION (CWE-78) - 5 examples ==========
    {
        "code": """import os
filename = request.args.get('file')
os.system(f'cat {filename}')""",
        "label": "vulnerable",
        "vulnerability_type": "command_injection",
        "severity": "critical",
        "cwe_id": "CWE-78",
        "description": "Command injection via os.system"
    },
    {
        "code": """import subprocess
user_input = get_user_input()
subprocess.call('ping ' + user_input, shell=True)""",
        "label": "vulnerable",
        "vulnerability_type": "command_injection",
        "severity": "critical",
        "cwe_id": "CWE-78",
        "description": "Command injection via subprocess with shell=True"
    },
    {
        "code": """import os
directory = request.form['dir']
os.popen(f'ls -la {directory}').read()""",
        "label": "vulnerable",
        "vulnerability_type": "command_injection",
        "severity": "high",
        "cwe_id": "CWE-78",
        "description": "Command injection via os.popen"
    },
    {
        "code": """from os import system
user_command = input('Enter command: ')
system(user_command)""",
        "label": "vulnerable",
        "vulnerability_type": "command_injection",
        "severity": "critical",
        "cwe_id": "CWE-78",
        "description": "Direct command execution from user input"
    },
    {
        "code": """import subprocess
filename = get_filename()
result = subprocess.run(f'grep pattern {filename}', shell=True, capture_output=True)""",
        "label": "vulnerable",
        "vulnerability_type": "command_injection",
        "severity": "high",
        "cwe_id": "CWE-78",
        "description": "Command injection in subprocess.run with shell=True"
    },

    # ========== PATH TRAVERSAL (CWE-22) - 3 examples ==========
    {
        "code": """filename = request.args.get('file')
with open('/data/files/' + filename) as f:
    content = f.read()""",
        "label": "vulnerable",
        "vulnerability_type": "path_traversal",
        "severity": "high",
        "cwe_id": "CWE-22",
        "description": "Path traversal via string concatenation"
    },
    {
        "code": """import os
user_file = request.form['filename']
filepath = os.path.join('/uploads', user_file)
os.remove(filepath)""",
        "label": "vulnerable",
        "vulnerability_type": "path_traversal",
        "severity": "high",
        "cwe_id": "CWE-22",
        "description": "Path traversal in file deletion"
    },
    {
        "code": """from pathlib import Path
user_path = get_user_input()
file_path = Path('/app/data') / user_path
content = file_path.read_text()""",
        "label": "vulnerable",
        "vulnerability_type": "path_traversal",
        "severity": "medium",
        "cwe_id": "CWE-22",
        "description": "Path traversal with pathlib without validation"
    },

    # ========== CODE INJECTION (CWE-94) - 5 examples ==========
    {
        "code": """user_code = request.form['code']
eval(user_code)""",
        "label": "vulnerable",
        "vulnerability_type": "code_injection",
        "severity": "critical",
        "cwe_id": "CWE-94",
        "description": "Arbitrary code execution via eval()"
    },
    {
        "code": """user_input = get_user_input()
exec(user_input)""",
        "label": "vulnerable",
        "vulnerability_type": "code_injection",
        "severity": "critical",
        "cwe_id": "CWE-94",
        "description": "Arbitrary code execution via exec()"
    },
    {
        "code": """expression = request.args.get('expr')
result = eval(expression, {"__builtins__": {}})""",
        "label": "vulnerable",
        "vulnerability_type": "code_injection",
        "severity": "high",
        "cwe_id": "CWE-94",
        "description": "Code injection via eval with restricted builtins (still unsafe)"
    },
    {
        "code": """import ast
user_code = request.data.decode()
exec(compile(user_code, '<string>', 'exec'))""",
        "label": "vulnerable",
        "vulnerability_type": "code_injection",
        "severity": "critical",
        "cwe_id": "CWE-94",
        "description": "Code injection via compile + exec"
    },
    {
        "code": """template = request.form['template']
result = eval(f'f"{template}"')""",
        "label": "vulnerable",
        "vulnerability_type": "code_injection",
        "severity": "high",
        "cwe_id": "CWE-94",
        "description": "Code injection via eval with f-string"
    },

    # ========== UNSAFE DESERIALIZATION (CWE-502) - 3 examples ==========
    {
        "code": """import pickle
data = request.data
obj = pickle.loads(data)""",
        "label": "vulnerable",
        "vulnerability_type": "unsafe_deserialization",
        "severity": "critical",
        "cwe_id": "CWE-502",
        "description": "Arbitrary code execution via pickle deserialization"
    },
    {
        "code": """import pickle
import base64

encoded_data = request.args.get('data')
decoded = base64.b64decode(encoded_data)
obj = pickle.loads(decoded)""",
        "label": "vulnerable",
        "vulnerability_type": "unsafe_deserialization",
        "severity": "critical",
        "cwe_id": "CWE-502",
        "description": "Pickle deserialization with base64 encoding"
    },
    {
        "code": """import yaml
config = yaml.load(user_input)""",
        "label": "vulnerable",
        "vulnerability_type": "unsafe_deserialization",
        "severity": "high",
        "cwe_id": "CWE-502",
        "description": "Unsafe YAML deserialization (use safe_load instead)"
    },

    # ========== HARDCODED CREDENTIALS (CWE-798) - 3 examples ==========
    {
        "code": """import psycopg2
conn = psycopg2.connect(
    host='localhost',
    database='mydb',
    user='admin',
    password='admin123'
)""",
        "label": "vulnerable",
        "vulnerability_type": "hardcoded_credentials",
        "severity": "high",
        "cwe_id": "CWE-798",
        "description": "Hardcoded database password"
    },
    {
        "code": """API_KEY = 'sk-1234567890abcdef'
headers = {'Authorization': f'Bearer {API_KEY}'}
response = requests.get(url, headers=headers)""",
        "label": "vulnerable",
        "vulnerability_type": "hardcoded_credentials",
        "severity": "high",
        "cwe_id": "CWE-798",
        "description": "Hardcoded API key"
    },
    {
        "code": """SECRET_KEY = 'my-secret-key-12345'
JWT_SECRET = 'jwt-secret-token'
app.config['SECRET_KEY'] = SECRET_KEY""",
        "label": "vulnerable",
        "vulnerability_type": "hardcoded_credentials",
        "severity": "medium",
        "cwe_id": "CWE-798",
        "description": "Hardcoded secret keys"
    },

    # ========== WEAK CRYPTOGRAPHY (CWE-327) - 3 examples ==========
    {
        "code": """import hashlib
password = user_input
hashed = hashlib.md5(password.encode()).hexdigest()""",
        "label": "vulnerable",
        "vulnerability_type": "weak_crypto",
        "severity": "medium",
        "cwe_id": "CWE-327",
        "description": "MD5 hash for passwords (cryptographically broken)"
    },
    {
        "code": """import hashlib
hash_obj = hashlib.sha1()
hash_obj.update(password.encode())
stored_hash = hash_obj.hexdigest()""",
        "label": "vulnerable",
        "vulnerability_type": "weak_crypto",
        "severity": "medium",
        "cwe_id": "CWE-327",
        "description": "SHA1 hash without salt"
    },
    {
        "code": """from Crypto.Cipher import DES
key = b'8bytekey'
cipher = DES.new(key, DES.MODE_ECB)
encrypted = cipher.encrypt(plaintext)""",
        "label": "vulnerable",
        "vulnerability_type": "weak_crypto",
        "severity": "high",
        "cwe_id": "CWE-327",
        "description": "DES encryption (deprecated, weak)"
    },

    # ========== SERVER-SIDE REQUEST FORGERY (CWE-918) - 2 examples ==========
    {
        "code": """import requests
url = request.args.get('url')
response = requests.get(url)
return response.content""",
        "label": "vulnerable",
        "vulnerability_type": "ssrf",
        "severity": "high",
        "cwe_id": "CWE-918",
        "description": "SSRF - unvalidated URL from user input"
    },
    {
        "code": """from urllib.request import urlopen
target_url = request.form['webhook_url']
data = urlopen(target_url).read()""",
        "label": "vulnerable",
        "vulnerability_type": "ssrf",
        "severity": "high",
        "cwe_id": "CWE-918",
        "description": "SSRF via urlopen with user-controlled URL"
    },
]


def _get_owasp_category(cwe_id):
    """Map CWE to OWASP Top 10 category"""
    mapping = {
        'CWE-89': 'A03:2021-Injection',
        'CWE-78': 'A03:2021-Injection',
        'CWE-94': 'A03:2021-Injection',
        'CWE-22': 'A01:2021-Broken Access Control',
        'CWE-502': 'A08:2021-Software and Data Integrity Failures',
        'CWE-798': 'A07:2021-Identification and Authentication Failures',
        'CWE-327': 'A02:2021-Cryptographic Failures',
        'CWE-918': 'A10:2021-Server-Side Request Forgery',
    }
    return mapping.get(cwe_id, 'Unknown')


# Add metadata
for i, pattern in enumerate(VULNERABLE_PATTERNS):
    pattern['id'] = f'vuln_{i + 1:03d}'
    if 'tags' not in pattern:
        pattern['tags'] = [pattern['vulnerability_type'], pattern['severity']]
    pattern['owasp_category'] = _get_owasp_category(pattern['cwe_id'])
