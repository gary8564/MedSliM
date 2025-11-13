#!/usr/bin/env bash
set -euo pipefail

# --- TRAINING DATA ---
FOLDER_URL='https://drive.google.com/drive/folders/0B0vscETPGI1-Q1h1WFdEM2FHSUE?resourcekey=0-XIVV_7YUjB9TPTQ3NfM17A'
DEST="/hpcwork/rwth1833/datasets/LiTS"   # destination on the cluster
REMOTE="gdrive"                           # your rclone remote name for Google Drive

mkdir -p "$DEST"

FOLDER_ID="$(sed -E 's#.*\/folders\/([^?]+).*#\1#' <<<"$FOLDER_URL")"
RESOURCE_KEY="$(sed -nE 's#.*[?&]resourcekey=([^&]+).*#\1#p' <<<"$FOLDER_URL")"

if ! rclone listremotes | grep -q "^${REMOTE}:" ; then
  echo "ERROR: rclone remote '$REMOTE' not found. Run 'rclone config' to create a Google Drive remote named '$REMOTE' first."
  exit 1
fi

echo "[info] Listing first items to verify access..."
rclone lsf "$REMOTE:" \
  --drive-root-folder-id "$FOLDER_ID" \
  ${RESOURCE_KEY:+--drive-resource-key "$RESOURCE_KEY"} \
  | head -n 20 || true

# Copy ONLY zip files (case-insensitive pattern list)
# --include limits to matching files; non-matching files are excluded.
echo "[info] Downloading only *.zip files to: $DEST"
rclone copy "$REMOTE:" "$DEST" \
  --drive-root-folder-id "$FOLDER_ID" \
  ${RESOURCE_KEY:+--drive-resource-key "$RESOURCE_KEY"} \
  --include "{*.zip,*.ZIP}" \
  --checksum \
  --fast-list \
  --transfers=8 --checkers=16 \
  --progress

# --- TEST DATA ---
FOLDER_URL='https://drive.google.com/drive/folders/0B0vscETPGI1-NDZNd3puMlZiNWM?resourcekey=0-dZUUwJiQnUVYVpRQvs_2tQ'
DEST="/hpcwork/rwth1833/datasets/LiTS/test_data"
REMOTE="gdrive"

mkdir -p "$DEST"

FOLDER_ID="$(sed -E 's#.*\/folders\/([^?]+).*#\1#' <<<"$FOLDER_URL")"
RESOURCE_KEY="$(sed -nE 's#.*[?&]resourcekey=([^&]+).*#\1#p' <<<"$FOLDER_URL")"

if [[ -z "${FOLDER_ID:-}" ]]; then
  echo "ERROR: Could not parse folder ID from URL." >&2
  exit 1
fi

if ! rclone listremotes | grep -q "^${REMOTE}:" ; then
  echo "ERROR: rclone remote '$REMOTE' not found. Run 'rclone config' to create it." >&2
  exit 1
fi

echo "[info] Verifying access to folder ID: $FOLDER_ID"
rclone lsf "${REMOTE}:" \
  --drive-root-folder-id "$FOLDER_ID" \
  ${RESOURCE_KEY:+--drive-resource-key "$RESOURCE_KEY"} \
  | head -n 20 || true

echo "[info] Downloading only NIfTI files (*.nii, *.nii.gz) to: $DEST"
rclone copy "${REMOTE}:" "$DEST" \
  --drive-root-folder-id "$FOLDER_ID" \
  ${RESOURCE_KEY:+--drive-resource-key "$RESOURCE_KEY"} \
  --include "*.[Nn][Ii][Ii]" \
  --include "*.[Nn][Ii][Ii].[Gg][Zz]" \
  --checksum \
  --fast-list \
  --transfers=8 \
  --checkers=16 \
  --progress

echo "Download completed. Data lives in: $DEST"