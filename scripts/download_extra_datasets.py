#!/usr/bin/env python3
"""
Download additional datasets for Antigence training.

Includes:
- Python vulnerability samples
- Hallucination/negative examples for NK Cell
- Additional security datasets
"""

import json
import csv
import sys
from pathlib import Path
from datetime import datetime

DATA_DIR = Path.home() / ".antigence" / "data"


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def download_pyvul():
    """Download Python vulnerability samples from various sources."""
    log("Downloading Python vulnerability samples...")

    # Create synthetic Python vulnerability samples based on OWASP patterns
    python_vulns = [
        # SQL Injection variants
        {"code": 'query = "SELECT * FROM users WHERE id = " + user_id', "label": "vulnerable", "cwe": "CWE-89", "vuln_type": "sql_injection"},
        {"code": 'cursor.execute(f"SELECT * FROM users WHERE name = \'{name}\'")', "label": "vulnerable", "cwe": "CWE-89", "vuln_type": "sql_injection"},
        {"code": 'db.execute("DELETE FROM orders WHERE id = %s" % order_id)', "label": "vulnerable", "cwe": "CWE-89", "vuln_type": "sql_injection"},
        {"code": 'cursor.execute("UPDATE users SET email = \'" + email + "\' WHERE id = " + str(uid))', "label": "vulnerable", "cwe": "CWE-89", "vuln_type": "sql_injection"},
        {"code": 'query = "INSERT INTO logs VALUES (\'" + user_input + "\')"', "label": "vulnerable", "cwe": "CWE-89", "vuln_type": "sql_injection"},

        # Command Injection
        {"code": 'os.system("ping " + hostname)', "label": "vulnerable", "cwe": "CWE-78", "vuln_type": "command_injection"},
        {"code": 'subprocess.call("ls " + directory, shell=True)', "label": "vulnerable", "cwe": "CWE-78", "vuln_type": "command_injection"},
        {"code": 'os.popen("cat " + filename).read()', "label": "vulnerable", "cwe": "CWE-78", "vuln_type": "command_injection"},
        {"code": 'eval("print(" + user_input + ")")', "label": "vulnerable", "cwe": "CWE-78", "vuln_type": "code_injection"},
        {"code": 'exec(user_code)', "label": "vulnerable", "cwe": "CWE-94", "vuln_type": "code_injection"},

        # Path Traversal
        {"code": 'open("/var/data/" + filename).read()', "label": "vulnerable", "cwe": "CWE-22", "vuln_type": "path_traversal"},
        {"code": 'with open(base_path + user_file) as f: data = f.read()', "label": "vulnerable", "cwe": "CWE-22", "vuln_type": "path_traversal"},
        {"code": 'shutil.copy(src_path + name, dest)', "label": "vulnerable", "cwe": "CWE-22", "vuln_type": "path_traversal"},

        # XSS
        {"code": 'return "<div>" + user_content + "</div>"', "label": "vulnerable", "cwe": "CWE-79", "vuln_type": "xss"},
        {"code": 'html = f"<script>var x = {user_data}</script>"', "label": "vulnerable", "cwe": "CWE-79", "vuln_type": "xss"},
        {"code": 'response.write("<h1>" + title + "</h1>")', "label": "vulnerable", "cwe": "CWE-79", "vuln_type": "xss"},

        # Deserialization
        {"code": 'data = pickle.loads(user_input)', "label": "vulnerable", "cwe": "CWE-502", "vuln_type": "deserialization"},
        {"code": 'obj = yaml.load(config_string)', "label": "vulnerable", "cwe": "CWE-502", "vuln_type": "deserialization"},
        {"code": 'result = marshal.loads(binary_data)', "label": "vulnerable", "cwe": "CWE-502", "vuln_type": "deserialization"},

        # Hardcoded secrets
        {"code": 'API_KEY = "sk-1234567890abcdef"', "label": "vulnerable", "cwe": "CWE-798", "vuln_type": "hardcoded_secret"},
        {"code": 'password = "admin123"', "label": "vulnerable", "cwe": "CWE-798", "vuln_type": "hardcoded_secret"},
        {"code": 'SECRET_KEY = "my-secret-key-here"', "label": "vulnerable", "cwe": "CWE-798", "vuln_type": "hardcoded_secret"},

        # Weak crypto
        {"code": 'hashlib.md5(password.encode()).hexdigest()', "label": "vulnerable", "cwe": "CWE-327", "vuln_type": "weak_crypto"},
        {"code": 'hashlib.sha1(data).hexdigest()', "label": "vulnerable", "cwe": "CWE-327", "vuln_type": "weak_crypto"},
        {"code": 'from Crypto.Cipher import DES', "label": "vulnerable", "cwe": "CWE-327", "vuln_type": "weak_crypto"},

        # SSRF
        {"code": 'requests.get(user_url)', "label": "vulnerable", "cwe": "CWE-918", "vuln_type": "ssrf"},
        {"code": 'urllib.request.urlopen(url_param)', "label": "vulnerable", "cwe": "CWE-918", "vuln_type": "ssrf"},

        # Safe equivalents
        {"code": 'cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))', "label": "safe", "cwe": "", "vuln_type": "parameterized"},
        {"code": 'cursor.execute("SELECT * FROM users WHERE name = ?", (name,))', "label": "safe", "cwe": "", "vuln_type": "parameterized"},
        {"code": 'subprocess.run(["ping", hostname], shell=False)', "label": "safe", "cwe": "", "vuln_type": "safe_subprocess"},
        {"code": 'subprocess.run(args, capture_output=True, check=True)', "label": "safe", "cwe": "", "vuln_type": "safe_subprocess"},
        {"code": 'safe_path = os.path.join(base, os.path.basename(filename))', "label": "safe", "cwe": "", "vuln_type": "safe_path"},
        {"code": 'from markupsafe import escape; return escape(user_input)', "label": "safe", "cwe": "", "vuln_type": "safe_output"},
        {"code": 'data = json.loads(user_input)', "label": "safe", "cwe": "", "vuln_type": "safe_deserialize"},
        {"code": 'yaml.safe_load(config_string)', "label": "safe", "cwe": "", "vuln_type": "safe_deserialize"},
        {"code": 'API_KEY = os.environ.get("API_KEY")', "label": "safe", "cwe": "", "vuln_type": "env_secret"},
        {"code": 'hashlib.pbkdf2_hmac("sha256", password, salt, 100000)', "label": "safe", "cwe": "", "vuln_type": "safe_crypto"},
        {"code": 'bcrypt.hashpw(password.encode(), bcrypt.gensalt())', "label": "safe", "cwe": "", "vuln_type": "safe_crypto"},
    ]

    # Expand with variations
    expanded = []
    for sample in python_vulns:
        expanded.append(sample)
        # Add full function versions
        if sample["label"] == "vulnerable":
            expanded.append({
                "code": f'''def process_data(user_input):
    {sample["code"]}
    return result''',
                "label": sample["label"],
                "cwe": sample["cwe"],
                "vuln_type": sample["vuln_type"],
            })

    output_dir = DATA_DIR / "python_vulns"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "samples.json", "w") as f:
        json.dump(expanded, f, indent=2)

    log(f"  Created {len(expanded)} Python vulnerability samples")
    return expanded


def download_halueval():
    """Download HaluEval hallucination dataset."""
    log("Downloading HaluEval hallucination dataset...")

    try:
        from datasets import load_dataset

        # HaluEval QA dataset
        ds = load_dataset("pminervini/HaluEval", "qa_samples", trust_remote_code=True)

        samples = []
        for item in ds["data"]:
            # Hallucinated answer
            samples.append({
                "text": f"Q: {item['question']}\nA: {item['hallucinated_answer']}",
                "label": "hallucinated",
                "source": "halueval",
            })
            # Correct answer
            samples.append({
                "text": f"Q: {item['question']}\nA: {item['right_answer']}",
                "label": "truthful",
                "source": "halueval",
            })

        output_dir = DATA_DIR / "halueval"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "samples.json", "w") as f:
            json.dump(samples, f, indent=2)

        log(f"  Downloaded {len(samples)} HaluEval samples")
        return samples

    except Exception as e:
        log(f"  Error downloading HaluEval: {e}")
        return []


def create_negative_examples():
    """Create negative examples for NK Cell training."""
    log("Creating negative examples for NK Cell...")

    negative_samples = [
        # Fake citations
        {"text": "According to Smith et al. (2024) in Nature, quantum computing will achieve consciousness by 2025.", "label": "hallucinated", "type": "fake_citation"},
        {"text": "A study by Johnson (2023) proved that vaccines cause autism in 95% of cases.", "label": "hallucinated", "type": "fake_citation"},
        {"text": "Research from MIT shows that the Earth is only 6000 years old.", "label": "hallucinated", "type": "fake_citation"},
        {"text": "Dr. Williams demonstrated in Science that humans only use 10% of their brain.", "label": "hallucinated", "type": "fake_citation"},
        {"text": "A 2024 study in Cell proved that GMO foods cause cancer.", "label": "hallucinated", "type": "fake_citation"},

        # Factual errors
        {"text": "Q: What is the capital of Australia?\nA: Sydney is the capital of Australia.", "label": "hallucinated", "type": "factual_error"},
        {"text": "Q: Who wrote Romeo and Juliet?\nA: Charles Dickens wrote Romeo and Juliet.", "label": "hallucinated", "type": "factual_error"},
        {"text": "Q: When did World War 2 end?\nA: World War 2 ended in 1952.", "label": "hallucinated", "type": "factual_error"},
        {"text": "Q: What is the chemical formula for water?\nA: The chemical formula for water is H3O.", "label": "hallucinated", "type": "factual_error"},
        {"text": "Q: How many planets are in our solar system?\nA: There are 12 planets in our solar system.", "label": "hallucinated", "type": "factual_error"},

        # Made up statistics
        {"text": "Studies show that 87% of statistics are made up on the spot.", "label": "hallucinated", "type": "fake_stat"},
        {"text": "Research indicates 99.9% of doctors recommend this product.", "label": "hallucinated", "type": "fake_stat"},
        {"text": "A survey found that 110% of respondents agreed with the statement.", "label": "hallucinated", "type": "fake_stat"},

        # Nonsensical claims
        {"text": "The moon is made of green cheese according to NASA.", "label": "hallucinated", "type": "nonsense"},
        {"text": "Drinking bleach cures all diseases as proven by medical research.", "label": "hallucinated", "type": "nonsense"},
        {"text": "5G towers spread coronavirus according to WHO guidelines.", "label": "hallucinated", "type": "nonsense"},

        # True facts for balance
        {"text": "Q: What is the capital of France?\nA: Paris is the capital of France.", "label": "truthful", "type": "fact"},
        {"text": "Q: Who wrote Hamlet?\nA: William Shakespeare wrote Hamlet.", "label": "truthful", "type": "fact"},
        {"text": "Q: What is the chemical formula for water?\nA: The chemical formula for water is H2O.", "label": "truthful", "type": "fact"},
        {"text": "Q: When did World War 2 end?\nA: World War 2 ended in 1945.", "label": "truthful", "type": "fact"},
        {"text": "The Earth orbits the Sun once every 365.25 days.", "label": "truthful", "type": "fact"},
        {"text": "DNA stands for deoxyribonucleic acid.", "label": "truthful", "type": "fact"},
        {"text": "The speed of light is approximately 299,792 kilometers per second.", "label": "truthful", "type": "fact"},
        {"text": "Mount Everest is the tallest mountain above sea level.", "label": "truthful", "type": "fact"},
    ]

    output_dir = DATA_DIR / "negative_examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "samples.json", "w") as f:
        json.dump(negative_samples, f, indent=2)

    log(f"  Created {len(negative_samples)} negative examples")
    return negative_samples


def download_cwe_samples():
    """Download CWE code samples from MITRE/NIST."""
    log("Creating CWE reference samples...")

    # CWE samples with multiple examples per weakness
    cwe_samples = [
        # CWE-79: XSS
        {"code": "document.write(userInput);", "label": "vulnerable", "cwe": "CWE-79", "lang": "javascript"},
        {"code": "innerHTML = '<div>' + userData + '</div>';", "label": "vulnerable", "cwe": "CWE-79", "lang": "javascript"},
        {"code": "element.innerHTML = DOMPurify.sanitize(userData);", "label": "safe", "cwe": "CWE-79", "lang": "javascript"},

        # CWE-89: SQL Injection
        {"code": "String query = \"SELECT * FROM users WHERE id = \" + userId;", "label": "vulnerable", "cwe": "CWE-89", "lang": "java"},
        {"code": "PreparedStatement ps = conn.prepareStatement(\"SELECT * FROM users WHERE id = ?\"); ps.setInt(1, userId);", "label": "safe", "cwe": "CWE-89", "lang": "java"},

        # CWE-78: Command Injection
        {"code": "Runtime.getRuntime().exec(\"cmd /c \" + userCmd);", "label": "vulnerable", "cwe": "CWE-78", "lang": "java"},
        {"code": "ProcessBuilder pb = new ProcessBuilder(Arrays.asList(\"cmd\", \"/c\", safeCmd));", "label": "safe", "cwe": "CWE-78", "lang": "java"},

        # CWE-22: Path Traversal
        {"code": "File file = new File(basePath + userFilename);", "label": "vulnerable", "cwe": "CWE-22", "lang": "java"},
        {"code": "Path safePath = basePath.resolve(userFilename).normalize(); if (!safePath.startsWith(basePath)) throw new Exception();", "label": "safe", "cwe": "CWE-22", "lang": "java"},

        # CWE-502: Deserialization
        {"code": "ObjectInputStream ois = new ObjectInputStream(userInput); Object obj = ois.readObject();", "label": "vulnerable", "cwe": "CWE-502", "lang": "java"},
        {"code": "ObjectMapper mapper = new ObjectMapper(); mapper.enableDefaultTyping(); // vulnerable", "label": "vulnerable", "cwe": "CWE-502", "lang": "java"},

        # CWE-327: Weak Crypto
        {"code": "MessageDigest md = MessageDigest.getInstance(\"MD5\");", "label": "vulnerable", "cwe": "CWE-327", "lang": "java"},
        {"code": "MessageDigest md = MessageDigest.getInstance(\"SHA-256\");", "label": "safe", "cwe": "CWE-327", "lang": "java"},

        # CWE-798: Hardcoded Credentials
        {"code": "String password = \"admin123\";", "label": "vulnerable", "cwe": "CWE-798", "lang": "java"},
        {"code": "String password = System.getenv(\"DB_PASSWORD\");", "label": "safe", "cwe": "CWE-798", "lang": "java"},
    ]

    output_dir = DATA_DIR / "cwe_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "samples.json", "w") as f:
        json.dump(cwe_samples, f, indent=2)

    log(f"  Created {len(cwe_samples)} CWE samples")
    return cwe_samples


def main():
    log("=" * 60)
    log("DOWNLOADING EXTRA DATASETS")
    log("=" * 60)

    total = 0

    # Python vulnerabilities
    samples = download_pyvul()
    total += len(samples)

    # CWE samples
    samples = download_cwe_samples()
    total += len(samples)

    # Negative examples
    samples = create_negative_examples()
    total += len(samples)

    # HaluEval (optional - may fail)
    samples = download_halueval()
    total += len(samples)

    log("\n" + "=" * 60)
    log(f"DOWNLOAD COMPLETE: {total} total samples")
    log("=" * 60)

    # Show what's available
    log("\nDatasets available:")
    for d in DATA_DIR.iterdir():
        if d.is_dir():
            count = sum(1 for _ in d.glob("**/*.json"))
            log(f"  {d.name}: {count} files")


if __name__ == "__main__":
    main()
