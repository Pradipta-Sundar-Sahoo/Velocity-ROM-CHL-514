#!/usr/bin/env bash
set -euo pipefail

RE_LIST=(725)

#
# Basilisk's qcc often isn't on PATH. We try PATH first, then fall back to
# the typical ~/basilisk/src/qcc location.
#
QCC_CMD=""
if command -v qcc >/dev/null 2>&1; then
  QCC_CMD="$(command -v qcc)"
else
  # In some WSL setups, $HOME can be mis-set (e.g. to a Windows path),
  # so don't rely on $HOME for Basilisk paths.
  for candidate in /home/*/basilisk/src/qcc; do
    if [[ -x "$candidate" ]]; then
      QCC_CMD="$candidate"
      break
    fi
  done
fi

if [[ -z "$QCC_CMD" ]]; then
  echo "Error: Basilisk qcc not found. Expected either 'qcc' on PATH or /home/*/basilisk/src/qcc."
  exit 1
fi

# Basilisk's headers are under the parent directory of the qcc executable.
# Example: /home/<user>/basilisk/src/qcc  => BASILISK=/home/<user>/basilisk/src
BASILISK_PATH="$(cd "$(dirname "$QCC_CMD")" && pwd)"
export BASILISK="$BASILISK_PATH"

export TMPDIR=/tmp

echo "Using qcc: $QCC_CMD"
echo "Using BASILISK: $BASILISK_PATH"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"
JOBS="${JOBS:-4}"
echo "Max parallel jobs: $JOBS"

if [[ ! -f "./simulation" && ! -f "./simulation.exe" ]]; then
  echo "Compiling simulation.c ..."
  "$QCC_CMD" -O2 -Wall ./simulation.c -o ./simulation -lm
fi

# Run multiple Reynolds numbers in parallel to speed up dataset generation.
for Re in "${RE_LIST[@]}"; do
  while [[ "$(jobs -rp | wc -l)" -ge "$JOBS" ]]; do
    sleep 1
  done

  (
    echo "Running Re=$Re ..."
    if ! ./simulation "$Re"; then
      echo "Warning: simulation failed for Re=$Re"
      exit 0
    fi

    out="steady_state_Re${Re}.csv"
    if [[ ! -s "$out" ]]; then
      echo "Warning: output missing/empty: $out"
    else
      echo "Generated $out"
    fi
  ) &
done

wait

echo "Done. Generated steady_state_Re*.csv in: $here"

