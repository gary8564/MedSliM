#!/usr/bin/bash

# Configuration
INPUT_DIR="/hpcwork/rwth1833/datasets/SKM-TEA/qdess/v1-release/dicoms"
OUTPUT_DIR="/hpcwork/rwth1833/datasets/SKM-TEA/qdess/v1-release/raw_images"

# Setup
mkdir -p "${OUTPUT_DIR}"

echo "======================================"
echo "SKM-TEA DICOM Extraction"
echo "======================================"
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Start time: $(date)"
echo ""

# Count total files
TOTAL_FILES=$(ls "${INPUT_DIR}"/*.tar.gz 2>/dev/null | wc -l)
echo "Total tar.gz files to extract: ${TOTAL_FILES}"
echo ""

# Counter for progress
COUNT=0

# Loop through all tar.gz files
for TAR_FILE in "${INPUT_DIR}"/*.tar.gz; do
    # Get the base name (e.g., MTR_001)
    BASENAME=$(basename "${TAR_FILE}" .tar.gz)
    
    # Increment counter
    COUNT=$((COUNT + 1))
    
    # Check if already extracted
    if [ -d "${OUTPUT_DIR}/${BASENAME}" ]; then
        echo "[${COUNT}/${TOTAL_FILES}] Skipping ${BASENAME} (already extracted)"
        continue
    fi
    
    echo "[${COUNT}/${TOTAL_FILES}] Extracting ${BASENAME}..."
    
    # Extract to output directory
    tar -xzf "${TAR_FILE}" -C "${OUTPUT_DIR}"
    
    # Check if extraction was successful
    if [ $? -eq 0 ]; then
        echo "  -> Successfully extracted ${BASENAME}"
    else
        echo "  -> ERROR extracting ${BASENAME}"
    fi
done

echo ""
echo "======================================"
echo "Extraction complete!"
echo "Total extracted: ${COUNT}"
echo "======================================"
