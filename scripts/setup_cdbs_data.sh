#!/usr/bin/env bash
# Relocate a pysynphot CDBS reference-file tree from a personal location
# (e.g. $HOME/Pysynphot/grp/redcat/trds/) to a shared, neutral location
# other users/accounts on the same machine can point PYSYN_CDBS at,
# instead of everyone needing their own multi-GB copy.
#
# Usage
# -----
#   ./setup_cdbs_data.sh <source_cdbs_dir> [dest_dir]
#
#   <source_cdbs_dir>  Existing CDBS tree, e.g. $HOME/Pysynphot/grp/redcat/trds/
#   [dest_dir]         Where to copy it to. Defaults to /usr/local/share/cdbs
#                       if writable, else /opt/pysynphot_cdbs
#                       (both are conventional, machine-wide, non-personal
#                       locations -- adjust for your site as needed, e.g.
#                       a path under shared NFS storage for a multi-machine
#                       cluster).
#
# What this does
# ---------------
# 1. Copies (rsync -a, so it's safe to re-run / resume) the CDBS tree to
#    the destination.
# 2. Prints the `export PYSYN_CDBS=...` line to add to /etc/environment,
#    a shared shell profile, or each user's own profile -- this script
#    does NOT edit any profile itself, since which file is appropriate
#    depends entirely on your site (single machine vs NFS-shared home
#    dirs vs a cluster with environment modules).
#
# Data source note
# -----------------
# This script only RELOCATES an existing CDBS tree; it does not download
# one. If you don't already have one, the reference files are published
# by STScI -- search for "CDBS pysynphot reference files download" for
# the current archive location, since these have moved hosts before.
set -euo pipefail

SRC="${1:?Usage: $0 <source_cdbs_dir> [dest_dir]}"
DEST="${2:-}"

if [ ! -d "$SRC" ]; then
    echo "ERROR: source directory '$SRC' does not exist" >&2
    exit 1
fi

if [ -z "$DEST" ]; then
    # This site's current shared CDBS location -- override with an
    # explicit [dest_dir] argument if relocating somewhere else.
    DEST="/home/phys/astronomy/Pysynphot_Files"
fi

echo "Copying CDBS tree:"
echo "  from: $SRC"
echo "  to:   $DEST"
mkdir -p "$DEST"
rsync -a --info=progress2 "$SRC"/ "$DEST"/

echo
echo "Done. Point PYSYN_CDBS at the new location, e.g. add this to a"
echo "shared shell profile (/etc/environment, /etc/profile.d/*.sh, or"
echo "each user's ~/.bashrc):"
echo
echo "    export PYSYN_CDBS=$DEST"
echo
echo "otehiwai_pouakai.config.pysyn_cdbs_dir() reads this same variable,"
echo "so no further code changes are needed once it's set."
