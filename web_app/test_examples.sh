#!/bin/bash
# Antigence Platform - Test Examples
# Run these commands to test the Publications feature

echo "========================================"
echo "Antigence Platform - Test Examples"
echo "========================================"
echo ""

echo "Test 1: Publications Validator (3 claims)"
echo "-------------------------------------------"
curl -s -X POST http://localhost:5001/api/validate_publications \
  -H "Content-Type: application/json" \
  -d '{"claims": ["Aspirin reduces the risk of cardiovascular disease.", "Vitamin D supplementation prevents COVID-19 infection.", "Regular exercise improves symptoms of depression."], "mode": "standard"}' \
  | python3 -m json.tool
echo ""

echo "Test 2: Single claim validation"
echo "-------------------------------------------"
curl -s -X POST http://localhost:5001/api/validate_publications \
  -H "Content-Type: application/json" \
  -d '{"claims": ["Smoking causes lung cancer."], "mode": "standard"}' \
  | python3 -m json.tool
echo ""

echo "Test 3: B Cell only mode"
echo "-------------------------------------------"
curl -s -X POST http://localhost:5001/api/validate_publications \
  -H "Content-Type: application/json" \
  -d '{"claims": ["Hydroxychloroquine is effective against SARS-CoV-2."], "mode": "bcell_only"}' \
  | python3 -m json.tool
echo ""

echo "========================================"
echo "All tests complete!"
echo "========================================"
echo ""
echo "Web UI available at: http://localhost:5001/publications"
echo ""
