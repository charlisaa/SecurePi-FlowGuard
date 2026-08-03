#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORT_DIR="${SCRIPT_DIR}/export"
OUTPUT_DIR="${SCRIPT_DIR}/rpk_output"
FINAL_MODEL="${SCRIPT_DIR}/imx500_custom_securepi.rpk"

if ! command -v imx500-package >/dev/null 2>&1; then
    echo "imx500-package is not installed."
    echo "Run: sudo apt update && sudo apt install -y imx500-tools imx500-all"
    exit 1
fi

if [[ ! -f "${EXPORT_DIR}/packerOut.zip" ]]; then
    echo "Missing ${EXPORT_DIR}/packerOut.zip"
    exit 1
fi

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

imx500-package           -i "${EXPORT_DIR}/packerOut.zip"           -o "${OUTPUT_DIR}"

if [[ ! -f "${OUTPUT_DIR}/network.rpk" ]]; then
    echo "Packaging completed but network.rpk was not found."
    exit 1
fi

mv "${OUTPUT_DIR}/network.rpk" "${FINAL_MODEL}"

echo
echo "SecurePi model installed:"
echo "${FINAL_MODEL}"
echo
echo "Run:"
echo "python3 edge/securePi.py @edge/presets/demo.args \\"
echo "  --model models/imx500_custom_securepi.rpk \\"
echo "  --labels models/labels.txt -v"
