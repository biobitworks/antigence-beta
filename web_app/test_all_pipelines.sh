#!/bin/bash
# Test all validation pipelines

echo "=========================================="
echo "Testing All Antigent Pipelines"
echo "=========================================="
echo ""

echo "1. ⚡ Quick Pipeline (B Cell only)"
echo "------------------------------------------"
curl -s -X POST http://localhost:5001/api/validate_publications \
  -H "Content-Type: application/json" \
  -d '{"claims": ["Aspirin reduces cardiovascular disease risk."], "mode": "quick"}' \
  | python3 -m json.tool
echo ""

echo "2. 🎯 Standard Pipeline (B Cell + NK Cell)"
echo "------------------------------------------"
curl -s -X POST http://localhost:5001/api/validate_publications \
  -H "Content-Type: application/json" \
  -d '{"claims": ["Smoking causes lung cancer."], "mode": "standard"}' \
  | python3 -m json.tool
echo ""

echo "3. 🔬 Enhanced Pipeline (+ Dendritic)"
echo "------------------------------------------"
curl -s -X POST http://localhost:5001/api/validate_publications \
  -H "Content-Type: application/json" \
  -d '{"claims": ["Exercise improves mental health."], "mode": "enhanced"}' \
  | python3 -m json.tool
echo ""

echo "4. 🧠 Deep Pipeline (+ Memory)"
echo "------------------------------------------"
curl -s -X POST http://localhost:5001/api/validate_publications \
  -H "Content-Type: application/json" \
  -d '{"claims": ["Vitamin D prevents COVID-19."], "mode": "deep"}' \
  | python3 -m json.tool
echo ""

echo "5. 🌐 Orchestrated Pipeline (+ LLM Thymus)"
echo "------------------------------------------"
curl -s -X POST http://localhost:5001/api/validate_publications \
  -H "Content-Type: application/json" \
  -d '{"claims": ["Hydroxychloroquine is effective against SARS-CoV-2."], "mode": "orchestrated"}' \
  | python3 -m json.tool
echo ""

echo "=========================================="
echo "Pipeline Comparison"
echo "=========================================="
echo "Quick:        B Cell only (~30ms)"
echo "Standard:     + NK Cell (~100ms)"
echo "Enhanced:     + Dendritic features (~200ms)"
echo "Deep:         + Memory search (~500ms)"
echo "Orchestrated: + LLM coordination (~2-5s)"
echo ""
