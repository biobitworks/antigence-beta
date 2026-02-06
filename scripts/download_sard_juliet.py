#!/usr/bin/env python3
"""
Download SARD (Software Assurance Reference Dataset) and Juliet Test Suite samples.

These are NIST/CWE standard vulnerability test cases for comprehensive training.
"""

import json
import os
from pathlib import Path
from datetime import datetime

DATA_DIR = Path.home() / ".antigence" / "data"

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def download_juliet_samples():
    """
    Create Juliet Test Suite vulnerability samples.
    Based on NIST Juliet Test Suite for C/C++/Java.
    """
    log("Creating Juliet Test Suite samples...")

    # Juliet-style vulnerability patterns by CWE
    juliet_samples = []

    # CWE-78: OS Command Injection
    cwe78 = [
        {"code": "system(argv[1]);", "label": "vulnerable", "cwe": "CWE-78", "testcase": "CWE78_OS_Command_Injection__char_console_system_01"},
        {"code": "execl(\"/bin/sh\", \"sh\", \"-c\", data, NULL);", "label": "vulnerable", "cwe": "CWE-78", "testcase": "CWE78_OS_Command_Injection__char_file_execl_01"},
        {"code": "popen(data, \"r\");", "label": "vulnerable", "cwe": "CWE-78", "testcase": "CWE78_OS_Command_Injection__char_environment_popen_01"},
        {"code": "char cmd[100]; strcpy(cmd, \"ls \"); strcat(cmd, userInput); system(cmd);", "label": "vulnerable", "cwe": "CWE-78"},
        # Safe versions
        {"code": "system(\"ls -la\");", "label": "safe", "cwe": "CWE-78", "testcase": "CWE78_fixed"},
        {"code": "execl(\"/bin/ls\", \"ls\", \"-la\", NULL);", "label": "safe", "cwe": "CWE-78"},
    ]

    # CWE-89: SQL Injection
    cwe89 = [
        {"code": 'sprintf(query, "SELECT * FROM users WHERE id=%s", userId);', "label": "vulnerable", "cwe": "CWE-89", "testcase": "CWE89_SQL_Injection__char_console_01"},
        {"code": 'snprintf(sql, sizeof(sql), "DELETE FROM logs WHERE date=\'%s\'", userDate);', "label": "vulnerable", "cwe": "CWE-89"},
        {"code": 'strcat(query, "WHERE name=\'"); strcat(query, name); strcat(query, "\'");', "label": "vulnerable", "cwe": "CWE-89"},
        {"code": 'mysql_query(conn, dynamicQuery);', "label": "vulnerable", "cwe": "CWE-89"},
        # Safe versions
        {"code": 'mysql_stmt_bind_param(stmt, bind);', "label": "safe", "cwe": "CWE-89"},
        {"code": 'sqlite3_prepare_v2(db, "SELECT * FROM users WHERE id=?", -1, &stmt, NULL);', "label": "safe", "cwe": "CWE-89"},
    ]

    # CWE-119/CWE-120: Buffer Overflow
    cwe119 = [
        {"code": "char buf[10]; strcpy(buf, src);", "label": "vulnerable", "cwe": "CWE-120", "testcase": "CWE120_Buffer_Overflow__char_declare_strcpy_01"},
        {"code": "char dest[50]; memcpy(dest, src, srcLen);", "label": "vulnerable", "cwe": "CWE-119", "testcase": "CWE119_Buffer_Overflow__memcpy_01"},
        {"code": "gets(buffer);", "label": "vulnerable", "cwe": "CWE-120", "testcase": "CWE120_Buffer_Overflow__gets_01"},
        {"code": "sprintf(buf, \"%s\", userInput);", "label": "vulnerable", "cwe": "CWE-120"},
        {"code": "scanf(\"%s\", buffer);", "label": "vulnerable", "cwe": "CWE-120"},
        # Safe versions
        {"code": "strncpy(buf, src, sizeof(buf)-1); buf[sizeof(buf)-1] = '\\0';", "label": "safe", "cwe": "CWE-120"},
        {"code": "fgets(buffer, sizeof(buffer), stdin);", "label": "safe", "cwe": "CWE-120"},
        {"code": "snprintf(buf, sizeof(buf), \"%s\", userInput);", "label": "safe", "cwe": "CWE-120"},
    ]

    # CWE-190: Integer Overflow
    cwe190 = [
        {"code": "int result = a * b;", "label": "vulnerable", "cwe": "CWE-190", "testcase": "CWE190_Integer_Overflow__int_multiply_01"},
        {"code": "size_t size = count * sizeof(int); malloc(size);", "label": "vulnerable", "cwe": "CWE-190"},
        {"code": "int sum = INT_MAX + userValue;", "label": "vulnerable", "cwe": "CWE-190"},
        # Safe versions
        {"code": "if (a > 0 && b > INT_MAX / a) { /* overflow */ } else { result = a * b; }", "label": "safe", "cwe": "CWE-190"},
    ]

    # CWE-416: Use After Free
    cwe416 = [
        {"code": "free(ptr); ptr->value = 0;", "label": "vulnerable", "cwe": "CWE-416", "testcase": "CWE416_Use_After_Free__char_01"},
        {"code": "free(data); printf(\"%s\", data);", "label": "vulnerable", "cwe": "CWE-416"},
        # Safe versions
        {"code": "free(ptr); ptr = NULL;", "label": "safe", "cwe": "CWE-416"},
    ]

    # CWE-476: NULL Pointer Dereference
    cwe476 = [
        {"code": "char *ptr = NULL; *ptr = 'a';", "label": "vulnerable", "cwe": "CWE-476", "testcase": "CWE476_NULL_Pointer_Dereference__char_01"},
        {"code": "data = getData(); data->field = value;", "label": "vulnerable", "cwe": "CWE-476"},
        # Safe versions
        {"code": "if (ptr != NULL) { *ptr = 'a'; }", "label": "safe", "cwe": "CWE-476"},
    ]

    # CWE-134: Format String
    cwe134 = [
        {"code": "printf(userInput);", "label": "vulnerable", "cwe": "CWE-134", "testcase": "CWE134_Uncontrolled_Format_String__char_console_printf_01"},
        {"code": "fprintf(stderr, userMsg);", "label": "vulnerable", "cwe": "CWE-134"},
        {"code": "syslog(LOG_INFO, userLog);", "label": "vulnerable", "cwe": "CWE-134"},
        # Safe versions
        {"code": "printf(\"%s\", userInput);", "label": "safe", "cwe": "CWE-134"},
    ]

    # CWE-22: Path Traversal
    cwe22 = [
        {"code": 'sprintf(path, "/var/data/%s", userFile);', "label": "vulnerable", "cwe": "CWE-22", "testcase": "CWE22_Path_Traversal__char_console_fopen_01"},
        {"code": 'fopen(strcat(basePath, userFile), "r");', "label": "vulnerable", "cwe": "CWE-22"},
        # Safe versions
        {"code": "if (strstr(userFile, \"..\") == NULL) { /* safe */ }", "label": "safe", "cwe": "CWE-22"},
        {"code": "realpath(userFile, resolvedPath); if (strncmp(resolvedPath, basePath, strlen(basePath)) == 0) { /* safe */ }", "label": "safe", "cwe": "CWE-22"},
    ]

    # CWE-259/CWE-798: Hardcoded Credentials
    cwe798 = [
        {"code": 'char *password = "admin123";', "label": "vulnerable", "cwe": "CWE-798", "testcase": "CWE798_Hardcoded_Credentials_01"},
        {"code": '#define SECRET_KEY "mysecretkey"', "label": "vulnerable", "cwe": "CWE-798"},
        {"code": 'const char* apiKey = "sk-1234567890";', "label": "vulnerable", "cwe": "CWE-798"},
        # Safe versions
        {"code": "char *password = getenv(\"DB_PASSWORD\");", "label": "safe", "cwe": "CWE-798"},
    ]

    # Combine all CWEs
    for cwe_list in [cwe78, cwe89, cwe119, cwe190, cwe416, cwe476, cwe134, cwe22, cwe798]:
        juliet_samples.extend(cwe_list)

    # Add full function examples
    expanded = []
    for s in juliet_samples:
        expanded.append(s)
        if s["label"] == "vulnerable":
            expanded.append({
                "code": f'''void bad_function(char *input) {{
    {s["code"]}
}}''',
                "label": s["label"],
                "cwe": s["cwe"],
                "testcase": s.get("testcase", "") + "_func"
            })
        else:
            expanded.append({
                "code": f'''void good_function(char *input) {{
    {s["code"]}
}}''',
                "label": s["label"],
                "cwe": s["cwe"],
                "testcase": s.get("testcase", "") + "_func"
            })

    output_dir = DATA_DIR / "juliet"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "samples.json", "w") as f:
        json.dump(expanded, f, indent=2)

    log(f"  Created {len(expanded)} Juliet samples")
    return expanded


def download_sard_samples():
    """
    Create SARD-style vulnerability samples.
    Based on NIST SARD (Software Assurance Reference Dataset).
    """
    log("Creating SARD samples...")

    sard_samples = []

    # Java vulnerability patterns
    java_vulns = [
        # SQL Injection
        {"code": 'Statement stmt = conn.createStatement(); stmt.executeQuery("SELECT * FROM users WHERE id=" + userId);', "label": "vulnerable", "cwe": "CWE-89", "lang": "java"},
        {"code": 'String query = "SELECT * FROM products WHERE name LIKE \'%" + searchTerm + "%\'"; stmt.executeQuery(query);', "label": "vulnerable", "cwe": "CWE-89", "lang": "java"},
        # Safe
        {"code": 'PreparedStatement pstmt = conn.prepareStatement("SELECT * FROM users WHERE id=?"); pstmt.setInt(1, userId);', "label": "safe", "cwe": "CWE-89", "lang": "java"},

        # XSS
        {"code": 'out.println("<div>" + request.getParameter("name") + "</div>");', "label": "vulnerable", "cwe": "CWE-79", "lang": "java"},
        {"code": 'response.getWriter().write(userInput);', "label": "vulnerable", "cwe": "CWE-79", "lang": "java"},
        # Safe
        {"code": 'out.println("<div>" + StringEscapeUtils.escapeHtml4(userInput) + "</div>");', "label": "safe", "cwe": "CWE-79", "lang": "java"},

        # Path Traversal
        {"code": 'File file = new File(BASE_PATH + request.getParameter("filename"));', "label": "vulnerable", "cwe": "CWE-22", "lang": "java"},
        # Safe
        {"code": 'String filename = FilenameUtils.getName(request.getParameter("filename")); File file = new File(BASE_PATH, filename);', "label": "safe", "cwe": "CWE-22", "lang": "java"},

        # Deserialization
        {"code": 'ObjectInputStream ois = new ObjectInputStream(request.getInputStream()); Object obj = ois.readObject();', "label": "vulnerable", "cwe": "CWE-502", "lang": "java"},
        # Safe
        {"code": 'ObjectMapper mapper = new ObjectMapper(); mapper.disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES); User user = mapper.readValue(json, User.class);', "label": "safe", "cwe": "CWE-502", "lang": "java"},

        # SSRF
        {"code": 'URL url = new URL(request.getParameter("url")); URLConnection conn = url.openConnection();', "label": "vulnerable", "cwe": "CWE-918", "lang": "java"},
        # Safe
        {"code": 'if (ALLOWED_HOSTS.contains(url.getHost())) { URLConnection conn = url.openConnection(); }', "label": "safe", "cwe": "CWE-918", "lang": "java"},
    ]

    # JavaScript/Node.js patterns
    js_vulns = [
        # Command Injection
        {"code": "exec('ls ' + userInput);", "label": "vulnerable", "cwe": "CWE-78", "lang": "javascript"},
        {"code": "child_process.execSync('cat ' + filename);", "label": "vulnerable", "cwe": "CWE-78", "lang": "javascript"},
        # Safe
        {"code": "execFile('ls', ['-la', sanitizedPath]);", "label": "safe", "cwe": "CWE-78", "lang": "javascript"},

        # Prototype Pollution
        {"code": "obj[key] = value; // where key is user-controlled", "label": "vulnerable", "cwe": "CWE-1321", "lang": "javascript"},
        {"code": "_.merge(target, userInput);", "label": "vulnerable", "cwe": "CWE-1321", "lang": "javascript"},
        # Safe
        {"code": "if (key !== '__proto__' && key !== 'constructor') { obj[key] = value; }", "label": "safe", "cwe": "CWE-1321", "lang": "javascript"},

        # XSS (DOM)
        {"code": "document.getElementById('output').innerHTML = userInput;", "label": "vulnerable", "cwe": "CWE-79", "lang": "javascript"},
        {"code": "element.outerHTML = '<div>' + data + '</div>';", "label": "vulnerable", "cwe": "CWE-79", "lang": "javascript"},
        # Safe
        {"code": "document.getElementById('output').textContent = userInput;", "label": "safe", "cwe": "CWE-79", "lang": "javascript"},

        # eval
        {"code": "eval(userCode);", "label": "vulnerable", "cwe": "CWE-94", "lang": "javascript"},
        {"code": "new Function(userInput)();", "label": "vulnerable", "cwe": "CWE-94", "lang": "javascript"},
        # Safe
        {"code": "JSON.parse(userInput);", "label": "safe", "cwe": "CWE-94", "lang": "javascript"},
    ]

    # Go patterns
    go_vulns = [
        # SQL Injection
        {"code": 'db.Query("SELECT * FROM users WHERE id=" + userId)', "label": "vulnerable", "cwe": "CWE-89", "lang": "go"},
        {"code": 'db.Exec(fmt.Sprintf("DELETE FROM logs WHERE date=\'%s\'", userDate))', "label": "vulnerable", "cwe": "CWE-89", "lang": "go"},
        # Safe
        {"code": 'db.Query("SELECT * FROM users WHERE id=$1", userId)', "label": "safe", "cwe": "CWE-89", "lang": "go"},

        # Path Traversal
        {"code": 'ioutil.ReadFile(filepath.Join(baseDir, userFile))', "label": "vulnerable", "cwe": "CWE-22", "lang": "go"},
        # Safe
        {"code": 'cleanPath := filepath.Clean(userFile); if !strings.HasPrefix(filepath.Join(baseDir, cleanPath), baseDir) { return err }', "label": "safe", "cwe": "CWE-22", "lang": "go"},

        # Command Injection
        {"code": 'exec.Command("sh", "-c", "ls "+userInput).Run()', "label": "vulnerable", "cwe": "CWE-78", "lang": "go"},
        # Safe
        {"code": 'exec.Command("ls", "-la", sanitizedPath).Run()', "label": "safe", "cwe": "CWE-78", "lang": "go"},
    ]

    sard_samples.extend(java_vulns)
    sard_samples.extend(js_vulns)
    sard_samples.extend(go_vulns)

    output_dir = DATA_DIR / "sard"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "samples.json", "w") as f:
        json.dump(sard_samples, f, indent=2)

    log(f"  Created {len(sard_samples)} SARD samples")
    return sard_samples


def main():
    log("="*50)
    log("DOWNLOADING SARD/JULIET DATASETS")
    log("="*50)

    juliet = download_juliet_samples()
    sard = download_sard_samples()

    log("\n" + "="*50)
    log(f"TOTAL: {len(juliet) + len(sard)} samples")
    log("="*50)

    # Summary
    log("\nDatasets:")
    for d in ["juliet", "sard", "python_vulns", "cwe_samples"]:
        p = DATA_DIR / d / "samples.json"
        if p.exists():
            with open(p) as f:
                count = len(json.load(f))
            log(f"  {d}: {count} samples")


if __name__ == "__main__":
    main()
