#!/bin/bash
# Antigence™ Data Setup Script
# Automates the acquisition of public datasets for Antigent training.

echo "--- Antigence™ Data Setup ---"

# 1. Create Directories
mkdir -p data/training/network data/training/security data/training/emotion

# 2. Network Data (NSL-KDD Sample)
echo "Fetching NSL-KDD subset..."
curl -L https://raw.githubusercontent.com/uoip/nsl-kdd/master/KDDTrain%2B.txt -o data/training/network/KDDTrain_plus.csv

# 3. Security Data (SQL Injection Sample)
echo "Fetching SQL Injection sample..."
curl -L https://raw.githubusercontent.com/streadway/sql-injections/master/sql-injections.txt -o data/training/security/sql_injections.txt

echo "--- Setup Complete ---"
echo "Datasets available in data/training/"
