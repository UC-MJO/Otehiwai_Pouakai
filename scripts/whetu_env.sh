#!/bin/sh
# Use after activating your Python environment:
#   source /path/to/Otehiwai_Pouakai/scripts/whetu_env.sh

export POUAKAI_RAW_ARCHIVE_DIR=/home/phys/astro8/MJArchive/octans
export POUAKAI_CAL_LIST_DIR=/home/phys/astronomy/Pouakai_cal_Lists
export POUAKAI_CAL_FILES_DIR=/home/phys/astronomy/Pouakai_cal_Files
export POUAKAI_MASTER_DARK_DIR=/home/phys/astronomy/Pouakai_Masters/Master_Darks
export POUAKAI_MASTER_FLAT_DIR=/home/phys/astronomy/Pouakai_Masters/Master_Flats
export PYSYN_CDBS=/home/phys/astronomy/Pysynphot_Files

# This selects the server's manual installation. To use an activated conda environment's
# solver instead, unset POUAKAI_ASTROMETRY_BIN.
export POUAKAI_ASTROMETRY_BIN=/usr/local/astrometry/bin
