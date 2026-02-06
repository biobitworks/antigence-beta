import csv
import os
import random


def generate_network_data(filename, count=200):
    """Generates synthetic network traffic features (NSL-KDD style)"""
    # Features: duration, protocol_type, service, flag, src_bytes, dst_bytes, label, class
    protocols = ["tcp", "udp", "icmp"]
    services = ["http", "smtp", "finger", "domain_u", "ftp_data"]
    flags = ["SF", "S0", "REJ", "RSTR"]

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "label"]
        )

        for _ in range(count):
            is_normal = random.random() > 0.2
            if is_normal:
                writer.writerow(
                    [
                        round(random.uniform(0, 1), 4),
                        random.choice(protocols),
                        random.choice(services),
                        "SF",
                        random.randint(100, 5000),
                        random.randint(100, 5000),
                        "normal",
                    ]
                )
            else:
                writer.writerow(
                    [
                        round(random.uniform(5, 50), 4),
                        "tcp",
                        "http",
                        "S0",
                        random.randint(5000, 100000),
                        0,
                        "anomaly",
                    ]
                )
    print(f"Generated {count} network patterns in {filename}")


def generate_security_data(filename, count=100):
    """Generates synthetic SQLi/XSS patterns"""
    safe_queries = [
        "SELECT * FROM users WHERE id = 1",
        "SELECT name, email FROM profiles WHERE active = 1",
        "INSERT INTO logs (msg) VALUES ('Login successful')",
        "UPDATE settings SET dark_mode = 1 WHERE user_id = 42",
    ]
    malicious_patterns = [
        "SELECT * FROM users WHERE id = 1 OR 1=1",
        "DROP TABLE users; --",
        "<script>alert('XSS')</script>",
        "' UNION SELECT username, password FROM users --",
    ]

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["content", "label"])
        for _ in range(count):
            is_normal = random.random() > 0.3
            if is_normal:
                writer.writerow([random.choice(safe_queries), "normal"])
            else:
                writer.writerow([random.choice(malicious_patterns), "malicious"])
    print(f"Generated {count} security patterns in {filename}")


if __name__ == "__main__":
    base_dir = "data/training"
    generate_network_data(os.path.join(base_dir, "network/synthetic_network.csv"))
    generate_security_data(os.path.join(base_dir, "security/synthetic_security.csv"))
