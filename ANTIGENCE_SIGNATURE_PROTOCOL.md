# Antigence Signature Protocol

**Effective**: 2026-01-05
**Required**: All commits to antigence-alpha must include Antigence review signature

## 🛡️ Protocol Overview

**Every commit MUST:**
1. Be reviewed by Antigence before pushing
2. Include Antigence security review signature in commit message
3. Include agent attribution (who made the changes)
4. Save review to `.immunos/reviews/` for audit trail

---

## 📋 Workflow

### 1. Make Changes
```bash
# Edit files normally
vim src/immunos_mcp/servers/simple_mcp_server.py
```

### 2. Stage Changes
```bash
git add <files>
```

### 3. Run Antigence Review
```bash
# Review staged changes
python3 scripts/antigence_review.py --staged --format full > /tmp/review.txt

# Check exit code
# 0 = LOW risk (safe)
# 1 = MEDIUM risk (caution)
# 2 = HIGH risk (requires manual review)
```

### 4. Create Commit with Signature
```bash
# Create commit message with review signature
git commit -m "Your commit title

Brief description of changes.

$(cat /tmp/review.txt)

Changes by:
  Agent Signature: [Model Name (model-id)]
  Session: [session-id]
  Status: [status]
  Signed-At: [ISO timestamp]
  Signature-ID: [sig_YYYYMMDD_HHMMSS_hash]

🤖 Generated with Claude Code

Co-Authored-By: [Primary Author]
Co-Authored-By: [Additional Authors if any]"
```

### 5. Push to GitHub
```bash
git push origin main
```

---

## 🔑 Required Signature Format

### Commit Message Structure

```
<Title: Brief summary>

<Body: Description of changes>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🛡️  ANTIGENCE SECURITY REVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Review ID: rev_YYYYMMDD_HHMMSS_hash
Timestamp: YYYY-MM-DDTHH:MM:SS
Risk Level: LOW|MEDIUM|HIGH|UNCERTAIN

Analysis Results:
  • Classification: [safe/vulnerable/N/A]
  • Confidence: XX.X%
  • Anomaly Detected: [YES/NO]
  • B Cell (Pattern): XX.X%
  • NK Cell (Anomaly): XX.X%

Security Assessment:
  🟢/🟡/🔴 [Risk explanation]

Agents Used: [agent list]
Diff Hash: [hash]

Verified by: Antigence Multi-Agent Security Scanner
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Changes by:
  • [File/Component]:
    Agent: [Name/Model]
    Signature: [Signature ID]
    Session: [Timestamp]

🤖 Generated with Claude Code

Co-Authored-By: [Authors]
```

---

## 👤 Agent Signatures

### For Claude Sonnet 4.5
```
Agent Signature: Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)
Session: YYYY-MM-DD-session-name
Status: <status>
Signed-At: YYYY-MM-DDTHH:MM:SSZ
Signature-ID: sig_YYYYMMDD_HHMMSS_hash
```

### For ChatGPT
```
Agent Signature: ChatGPT o1-pro (<model-id>)
Session: YYYY-MM-DD-session-name
Status: <status>
Signed-At: YYYY-MM-DDTHH:MM:SSZ
Signature-ID: sig_YYYYMMDD_HHMMSS_hash
```

### For Qwen Coder
```
Agent Signature: Qwen 2.5 Coder 7B (qwen2.5-coder:7b)
Session: YYYY-MM-DD-session-name
Status: <status>
Signed-At: YYYY-MM-DDTHH:MM:SSZ
Signature-ID: sig_YYYYMMDD_HHMMSS_hash
```

### For DeepSeek R1
```
Agent Signature: DeepSeek R1 14B (deepseek-r1:14b)
Session: YYYY-MM-DD-session-name
Status: <status>
Signed-At: YYYY-MM-DDTHH:MM:SSZ
Signature-ID: sig_YYYYMMDD_HHMMSS_hash
```

---

## 📁 Review Storage

**Location**: `.immunos/reviews/`

**Filename**: `rev_YYYYMMDD_HHMMSS_<hash>.json`

**Format**:
```json
{
  "review_id": "rev_20260105_203706_39c8756b",
  "timestamp": "2026-01-05T20:37:06.478147",
  "description": "Brief description",
  "analysis": {
    "classification": "safe",
    "confidence": 0.85,
    "anomaly": false,
    "bcell_confidence": 0.90,
    "nk_confidence": 0.20
  },
  "risk_assessment": "LOW",
  "signals": {...},
  "agents_used": [...],
  "diff_hash": "abc123"
}
```

**Retention**: Keep all review files for audit trail (tracked in git)

---

## 🚨 Risk Level Interpretation

### 🟢 LOW RISK
- **Action**: Safe to commit and push
- **Meaning**: No security issues, standard changes
- **Example**: Documentation updates, refactoring

### 🟡 MEDIUM RISK
- **Action**: Proceed with caution, review changes
- **Meaning**: Systematic changes detected, no threats found
- **Example**: Renaming, structural changes, API updates

### 🔴 HIGH RISK
- **Action**: Manual security review required
- **Meaning**: Potential security issues detected
- **Example**: Security-critical code changes, external dependencies

### 🟡 UNCERTAIN
- **Action**: Manual review recommended
- **Meaning**: Low confidence in classification
- **Example**: Complex changes, insufficient context

---

## 🎯 Examples

### Example 1: Low Risk Commit

```bash
git add docs/README.md
python3 scripts/antigence_review.py --staged --format full > /tmp/review.txt

git commit -m "Update README with installation instructions

Added step-by-step installation guide for new users.

$(cat /tmp/review.txt)

Changes by:
  Agent: Claude Sonnet 4.5
  Signature: Sentinel A-20260105-01
  Session: 2026-01-05T20:00:00Z

🤖 Generated with Claude Code

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

git push origin main
```

### Example 2: Medium Risk Commit (Systematic Changes)

```bash
git add src/immunos_mcp/servers/simple_mcp_server.py
python3 scripts/antigence_review.py --staged --format full > /tmp/review.txt

# Review shows MEDIUM risk (systematic renaming)
# Still safe to commit after verification

git commit -m "Rename MCP tools to antigence_* namespace

BREAKING: Tool names changed for UX improvement.

$(cat /tmp/review.txt)

Changes by:
  Agent: Claude Sonnet 4.5
  Signature: Sentinel A-20260105-02
  Session: 2026-01-05T21:00:00Z

🤖 Generated with Claude Code

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

git push origin main
```

### Example 3: High Risk (Requires Review)

```bash
git add src/immunos_mcp/security/auth.py
python3 scripts/antigence_review.py --staged --format full

# Exit code 2 - HIGH RISK
# DO NOT commit yet
# Review changes manually
# If safe after review, add manual approval note
```

---

## 🔧 Automation

### Pre-Commit Hook (Optional)

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "🔍 Running Antigence security review..."

# Run review
python3 scripts/antigence_review.py --staged --format compact

EXIT_CODE=$?

if [ $EXIT_CODE -eq 2 ]; then
    echo "🔴 HIGH RISK - Commit blocked"
    echo "Run manual review: python3 scripts/antigence_review.py --staged --format full"
    exit 1
elif [ $EXIT_CODE -eq 1 ]; then
    echo "🟡 MEDIUM RISK - Review recommended"
    read -p "Continue with commit? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✅ Antigence review passed"
exit 0
```

---

## 📚 KB Update Required

**Add to Knowledge Base**:

1. **Commit Protocol**: All commits require Antigence review
2. **Signature Format**: Standard format for all agents
3. **Review Storage**: `.immunos/reviews/` tracked in git
4. **Agent Attribution**: Every change must include agent signature
5. **Risk Assessment**: Follow risk-based workflow

**Files to Update**:
- `docs/CONTRIBUTING.md`
- `docs/DEVELOPMENT.md`
- `README.md` (add badge: 🛡️ Antigence-Reviewed)

---

## ✅ Checklist Before Push

- [ ] Changes staged (`git add`)
- [ ] Antigence review run (`antigence_review.py --staged`)
- [ ] Review signature copied to commit message
- [ ] Agent attribution included
- [ ] Risk level acceptable for push
- [ ] Review saved to `.immunos/reviews/`
- [ ] Co-authors listed (if applicable)

---

## 🔧 Troubleshooting

### GPG Signing Issues

If `immunos_commit.sh` fails with GPG errors:

```bash
error: cannot run gpg: No such file or directory
fatal: failed to write commit object
```

**Solutions:**

1. **Configure SSH signing** (recommended for agent-specific keys):
```bash
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub
```

2. **Disable signing temporarily** (for testing):
```bash
# Modify immunos_commit.sh line 153 to remove -S flag:
git commit -F "$TMPFILE" $ARGS  # instead of: git commit -S -F "$TMPFILE" $ARGS
```

3. **Skip signing for one commit**:
```bash
git commit --no-gpg-sign -F <message-file>
```

### Bypassing Git Hooks

If git hooks block legitimate commits:

```bash
# Emergency bypass (use sparingly):
git commit --no-verify -m "Message"

# Or set environment variable:
export IMMUNOS_SKIP_HOOKS=1
git commit -m "Message"
```

**Note**: Bypassing hooks skips trailer validation. Ensure commit message includes all required trailers manually.

### Required Trailers Checklist

For manual commits without `immunos_commit.sh`, include:

```
Agent-Type: <type>
Agent-ID: <id>
Session-ID: <session>
Session-Started-At: <timestamp>
Key-ID: <key>
IMMUNOS-Trust: <trust level>
Model: <model-id>
Agent-Signature: <signature>
Agent-Status: <status>
Antigence-Review-ID: <review-id>
Antigence-Review-Risk: <LOW|MEDIUM|HIGH|UNCERTAIN>
Antigence-Review-Confidence: <0.xx>
Antigence-Review-Diff-Hash: <hash>
Antigence-Review-Timestamp: <timestamp>
```

Get session trailers:
```bash
python3 scripts/immunos_agent_identity.py trailers
```

---

## 🔮 Future Enhancements

1. **GitHub Actions**: Automatic Antigence review on PRs
2. **Review Dashboard**: Web UI to browse all reviews
3. **Trend Analysis**: Track risk levels over time
4. **Badge System**: Display review status in README
5. **API Integration**: Antigence review via GitHub bot

---

**Antigence Signature Protocol**
*Security verification for every commit*
*Effective: 2026-01-05*
