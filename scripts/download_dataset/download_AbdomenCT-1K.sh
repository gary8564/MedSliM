#!/usr/bin/env bash
# 
# Usage:
#   ./download_AbdomenCT-1K.sh [--out DIR] [RECORD_URL_OR_ID ...]

set -Eeuo pipefail

SCRIPT_NAME=$(basename "$0")

DEFAULT_OUTPUT_DIR="/hpcwork/rwth1833/datasets/AbdomenCT-1K"
# Default records provided in request
DEFAULT_RECORDS=(
  "https://zenodo.org/records/5903099"
  "https://zenodo.org/records/5903846"
  "https://zenodo.org/records/5903769"
)

log() { printf '[%s] %s\n' "$SCRIPT_NAME" "$*"; }
err() { printf '[%s][ERROR] %s\n' "$SCRIPT_NAME" "$*" >&2; }

require_one_of() {
  # usage: require_one_of cmd1 [cmd2 ...]
  for c in "$@"; do
    if command -v "$c" >/dev/null 2>&1; then
      return 0
    fi
  done
  err "Missing required tool. Install one of: $*"
  exit 1
}

extract_record_id() {
  # Accepts a Zenodo URL or numeric ID and prints the numeric ID
  local input=${1:-}
  if [[ -z "$input" ]]; then
    err "Empty record identifier"
    return 1
  fi
  if [[ "$input" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$input"
    return 0
  fi
  if [[ "$input" =~ zenodo\.org/records/([0-9]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  err "Unrecognized record identifier: $input"
  return 1
}

choose_downloader() {
  # Prefer wget if available; else curl
  if command -v wget >/dev/null 2>&1; then
    echo wget
  elif command -v curl >/dev/null 2>&1; then
    echo curl
  else
    err "Neither wget nor curl is installed. Please install one of them."
    exit 1
  fi
}

download_file() {
  # usage: download_file URL DEST_PATH
  local url=$1
  local dest=$2
  local downloader
  downloader=$(choose_downloader)
  if [[ "$downloader" == wget ]]; then
    wget -c -O "$dest" "$url"
  else
    # curl with resume, retries and follow redirects
    curl -fL --retry 5 --retry-delay 5 -C - -o "$dest" "$url"
  fi
}

compute_checksum() {
  # usage: compute_checksum ALGO FILE
  local algo=$1
  local file=$2
  case "$algo" in
    md5)
      if command -v md5sum >/dev/null 2>&1; then md5sum "$file" | awk '{print $1}'; else return 127; fi
      ;;
    sha256)
      if command -v sha256sum >/dev/null 2>&1; then sha256sum "$file" | awk '{print $1}'; else return 127; fi
      ;;
    sha1)
      if command -v sha1sum >/dev/null 2>&1; then sha1sum "$file" | awk '{print $1}'; else return 127; fi
      ;;
    *)
      return 2
      ;;
  esac
}

verify_checksum() {
  # usage: verify_checksum CHECKSUM_STRING FILE
  # CHECKSUM_STRING is like "md5:abcd..." or "sha256:abcd..."
  local checksum=$1
  local file=$2
  if [[ -z "$checksum" ]]; then
    log "No checksum provided by Zenodo for $(basename "$file"); skipping verification"
    return 0
  fi
  local algo value
  algo=${checksum%%:*}
  value=${checksum#*:}
  if [[ -z "$algo" || -z "$value" ]]; then
    log "Malformed checksum '$checksum' for $(basename "$file"); skipping verification"
    return 0
  fi
  local computed
  if ! computed=$(compute_checksum "$algo" "$file" 2>/dev/null); then
    log "Checksum tool for '$algo' not available; skipping verification for $(basename "$file")"
    return 0
  fi
  if [[ "$computed" == "$value" ]]; then
    log "Checksum OK ($algo) for $(basename "$file")"
  else
    err "Checksum MISMATCH for $(basename "$file"): expected $value got $computed"
    return 1
  fi
}

extract_files_with_jq() {
  # usage: extract_files_with_jq JSON
  # prints: name<TAB>checksum<TAB>download_url<TAB>size
  jq -r '
    .files[] |
    [
      (.key // .name // ""),
      (.checksum // ""),
      (.links.download // .links.self // ""),
      ((.size // 0) | tostring)
    ] | @tsv
  '
}

extract_files_with_python() {
  # usage: extract_files_with_python JSON
  # prints: name<TAB>checksum<TAB>download_url<TAB>size
  python3 - "$@" << 'PY'
import json,sys
data=json.load(sys.stdin)
files=data.get('files') or []
for f in files:
    name = f.get('key') or f.get('name') or ''
    checksum = f.get('checksum') or ''
    links = f.get('links') or {}
    url = links.get('download') or links.get('self') or ''
    size = f.get('size') or 0
    print(f"{name}\t{checksum}\t{url}\t{size}")
PY
}

list_record_files() {
  # usage: list_record_files RECORD_ID
  # outputs TSV rows: name<TAB>checksum<TAB>download_url<TAB>size
  local rid=$1
  local api_url="https://zenodo.org/api/records/${rid}"
  local json
  json=$(curl -sSL "$api_url")
  if [[ -z "$json" || "$json" == "null" ]]; then
    err "Failed to fetch metadata for record $rid"
    return 1
  fi
  if command -v jq >/dev/null 2>&1; then
    awk '1' <<< "$json" | extract_files_with_jq
  else
    if command -v python3 >/dev/null 2>&1; then
      awk '1' <<< "$json" | extract_files_with_python
    else
      err "Install either 'jq' or 'python3' to parse Zenodo API responses."
      return 1
    fi
  fi
}

download_record() {
  # usage: download_record RECORD_ID OUTPUT_DIR
  local rid=$1
  local out_dir=$2
  local record_dir="$out_dir/zenodo_${rid}"
  mkdir -p "$record_dir"
  log "Fetching file list for record $rid"
  local tsv
  if ! tsv=$(list_record_files "$rid"); then
    err "Skipping record $rid due to metadata fetch error"
    return 1
  fi
  local count=0
  while IFS=$'\t' read -r name checksum url size; do
    [[ -z "$name" || -z "$url" ]] && continue
    count=$((count+1))
    local dest="$record_dir/$name"
    mkdir -p "$(dirname "$dest")"
    log "Downloading ($count): $name ($size bytes)"
    download_file "$url" "$dest"
    verify_checksum "$checksum" "$dest" || true
  done <<< "$tsv"
  if [[ $count -eq 0 ]]; then
    err "No files found for record $rid"
    return 1
  fi
  log "Finished record $rid ($count file(s))"
}

print_usage() {
  cat <<USAGE
Usage: $SCRIPT_NAME [--out DIR] [RECORD_URL_OR_ID ...]

Downloads all files for the specified Zenodo record(s).

Options:
  --out DIR     Output directory (default: $DEFAULT_OUTPUT_DIR)

Arguments:
  RECORD_URL_OR_ID  One or more Zenodo record URLs or numeric IDs.
                    If omitted, uses the built-in records from the request.

Examples:
  $SCRIPT_NAME
  $SCRIPT_NAME --out /path/to/storage 5903099 5903846 5903769
USAGE
}

main() {
  local out_dir="$DEFAULT_OUTPUT_DIR"
  local records=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        print_usage; exit 0 ;;
      --out)
        shift; out_dir=${1:-}; [[ -z "$out_dir" ]] && { err "--out requires a directory"; exit 1; } ;;
      --)
        shift; break ;;
      -*)
        err "Unknown option: $1"; print_usage; exit 1 ;;
      *)
        records+=("$1") ;;
    esac
    shift || true
  done

  # Consume any remaining positional args as records
  while [[ $# -gt 0 ]]; do
    records+=("$1"); shift || true
  done

  mkdir -p "$out_dir"

  # Determine records to process
  if [[ ${#records[@]} -eq 0 ]]; then
    records=("${DEFAULT_RECORDS[@]}")
  fi

  # Process each record
  for rec in "${records[@]}"; do
    if ! rid=$(extract_record_id "$rec"); then
      err "Skipping invalid record identifier: $rec"
      continue
    fi
    download_record "$rid" "$out_dir"
  done

  log "All done. Data saved under: $out_dir"
}

main "$@"


