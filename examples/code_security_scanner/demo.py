"""
Interactive Code Security Scanner Demo

Demonstrates the IMMUNOS-MCP Code Security Scanner in action.
Users can scan code snippets interactively or run predefined examples.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scanner import CodeSecurityScanner


def print_banner():
    """Print demo banner"""
    print("\n" + "=" * 70)
    print("🛡️  IMMUNOS-MCP CODE SECURITY SCANNER")
    print("=" * 70)
    print("Artificial Immune System for Vulnerability Detection")
    print("B Cell (Pattern Matching) + NK Cell (Anomaly Detection)")
    print("=" * 70 + "\n")


def print_result(result):
    """Pretty print scan result"""
    # Risk level with color indicators
    risk_symbols = {
        'low': '✅',
        'medium': '⚠️ ',
        'high': '🚨',
        'critical': '🔴'
    }

    print("\n" + "─" * 70)
    print(f"{risk_symbols.get(result.risk_level, '❓')} RISK LEVEL: {result.risk_level.upper()}")
    print(f"   Vulnerable: {'YES' if result.is_vulnerable else 'NO'}")
    print(f"   Confidence: {result.confidence:.1%}")
    print("─" * 70)

    # Detailed explanation
    print("\n" + result.explanation)

    # Vulnerability types
    if result.vulnerability_types:
        print(f"\n📋 Vulnerability Types Detected:")
        for vuln in result.vulnerability_types:
            print(f"   • {vuln.replace('_', ' ').title()}")

    # Recommendations
    if result.recommendations:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"   {i}. {rec}")

    # Technical details
    print(f"\n🔬 Technical Details:")
    print(f"   B Cell Classification: {result.bcell_result.predicted_class}")
    print(f"   B Cell Confidence: {result.bcell_result.confidence:.1%}")
    print(f"   NK Cell Anomaly Score: {result.nk_cell_result.anomaly_score:.3f}")
    print(f"   NK Cell Is Anomaly: {result.nk_cell_result.is_anomaly}")

    print("\n" + "─" * 70 + "\n")


def run_predefined_examples(scanner):
    """Run predefined vulnerability examples"""
    examples = [
        {
            "name": "SQL Injection (F-String)",
            "code": """user_id = request.args.get('id')
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)""",
            "description": "Vulnerable: Uses f-string formatting in SQL query"
        },
        {
            "name": "Safe Parameterized Query",
            "code": """user_id = request.args.get('id')
cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))""",
            "description": "Safe: Uses parameterized query with placeholders"
        },
        {
            "name": "Command Injection",
            "code": """import os
filename = request.args.get('file')
os.system(f'cat {filename}')""",
            "description": "Vulnerable: Executes user input as system command"
        },
        {
            "name": "Safe Subprocess Usage",
            "code": """import subprocess
filename = validate_filename(request.args.get('file'))
subprocess.run(['cat', filename], check=True)""",
            "description": "Safe: Uses subprocess with list arguments (no shell)"
        },
        {
            "name": "Hardcoded Credentials",
            "code": """import psycopg2
conn = psycopg2.connect(
    host='localhost',
    database='mydb',
    user='admin',
    password='admin123'
)""",
            "description": "Vulnerable: Database password hardcoded in source"
        },
        {
            "name": "Environment Variable Credentials",
            "code": """import psycopg2
import os
conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)""",
            "description": "Safe: Credentials loaded from environment variables"
        },
        {
            "name": "Code Injection (eval)",
            "code": """user_code = request.form['code']
result = eval(user_code)""",
            "description": "Vulnerable: Executes arbitrary user code with eval()"
        },
        {
            "name": "Weak Cryptography (MD5)",
            "code": """import hashlib
password = user_input
hashed = hashlib.md5(password.encode()).hexdigest()""",
            "description": "Vulnerable: Uses MD5 for password hashing"
        },
    ]

    print("\n" + "=" * 70)
    print("PREDEFINED VULNERABILITY EXAMPLES")
    print("=" * 70)

    for i, example in enumerate(examples, 1):
        print(f"\n[{i}/{len(examples)}] {example['name']}")
        print(f"Description: {example['description']}")
        print(f"\nCode:")
        print("─" * 70)
        print(example['code'])
        print("─" * 70)

        # Scan the code
        result = scanner.scan_code(example['code'])
        print_result(result)

        # Wait for user to continue (except on last example)
        if i < len(examples):
            input("Press Enter to continue to next example...")


def run_interactive_mode(scanner):
    """Interactive code scanning mode"""
    print("\n" + "=" * 70)
    print("INTERACTIVE SCANNING MODE")
    print("=" * 70)
    print("Enter Python code to scan (type 'END' on a new line when done)")
    print("Commands: 'examples' = run predefined examples, 'quit' = exit")
    print("=" * 70 + "\n")

    while True:
        print("Enter code to scan (or command):")
        lines = []
        while True:
            try:
                line = input()
                if line.strip().lower() == 'end':
                    break
                elif line.strip().lower() == 'quit':
                    print("\n👋 Goodbye!")
                    return False
                elif line.strip().lower() == 'examples':
                    return True  # Signal to run examples
                lines.append(line)
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Goodbye!")
                return False

        if not lines:
            print("No code entered. Try again.\n")
            continue

        code = '\n'.join(lines)

        # Scan the code
        print("\n🔍 Scanning code...")
        result = scanner.scan_code(code)
        print_result(result)


def main():
    """Main demo function"""
    print_banner()

    # Initialize and train scanner
    print("🧬 Initializing Code Security Scanner...")
    scanner = CodeSecurityScanner(use_enhanced_nk=True)

    print("📚 Training on vulnerability patterns...")
    scanner.train()

    print("\n✅ Scanner ready!")
    stats = scanner.get_statistics()
    print(f"   • B Cell patterns: {stats['bcell_patterns']} classes")
    print(f"   • NK Cell detectors: {stats['nk_detectors']}")
    print(f"   • Embedding dimensions: {stats['embedding_dim']}")
    print(f"   • Using Enhanced NK Cell: {stats['using_enhanced_nk']}")

    # Main menu
    while True:
        print("\n" + "=" * 70)
        print("MAIN MENU")
        print("=" * 70)
        print("1. Run predefined examples")
        print("2. Interactive scanning mode")
        print("3. Quick test (SQL injection)")
        print("4. Exit")
        print("=" * 70)

        choice = input("\nSelect option (1-4): ").strip()

        if choice == '1':
            run_predefined_examples(scanner)

        elif choice == '2':
            run_examples = run_interactive_mode(scanner)
            if run_examples:
                run_predefined_examples(scanner)

        elif choice == '3':
            # Quick test
            test_code = """user_id = request.args.get('id')
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)"""

            print("\n🔍 Scanning SQL injection example...")
            result = scanner.scan_code(test_code)
            print_result(result)

        elif choice == '4':
            print("\n👋 Thank you for using IMMUNOS-MCP Code Security Scanner!")
            break

        else:
            print("\n❌ Invalid choice. Please select 1-4.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
