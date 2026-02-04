"""
supertem.vendor.JEOL.jeol_eos_tables

Default optical tables for JEOL microscopes.
Primary Reference: JEOL F200 (User Provided).

Structure: { MODE_KEY: { "MagList": [...], "CamList": [...] } }
"""

# =============================================================================
# 1. TEM MODES
# =============================================================================

# TEM:MAG
# Reference: F200
# MagList: Magnification (Unit: x)
TEM_MAG = {
    "MagList": [
        1000, 1200, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000,
        10000, 12000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 80000,
        100000, 120000, 150000, 200000, 250000, 300000, 400000, 500000, 600000, 800000,
        1000000, 1200000, 1500000, 2000000
    ],
    "CamList": []
}

# TEM:MAG2
# Reference: PyJEM Offline / Fallback
# MagList: Magnification (Unit: x)
TEM_MAG2 = {
    "MagList": [
        200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000, 2500, 3000,
        4000, 5000, 6000, 8000, 10000, 12000, 15000, 20000, 25000, 30000,
        40000, 50000, 60000, 80000, 100000, 120000, 150000, 200000, 250000,
        300000, 500000, 600000, 800000, 1000000, 1200000
    ],
    "CamList": []
}

# TEM:LOWMAG
# Reference: F200
# MagList: Magnification (Unit: x)
TEM_LOWMAG = {
    "MagList": [
        20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500,
        600, 800, 1000, 1200, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000,
        10000, 12000, 15000, 20000, 25000, 30000, 40000, 50000, 60000
    ],
    "CamList": []
}

# TEM:SAMAG (Selected Area Mag)
# Reference: PyJEM Offline
# MagList: Magnification (Unit: x)
TEM_SAMAG = {
    "MagList": [
        2000, 2500, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000,
        20000, 25000, 30000, 40000, 50000, 60000, 80000, 100000, 120000,
        150000, 200000, 250000, 300000
    ],
    "CamList": []
}

# TEM:DIFF
# Reference: F200
# MagList: Camera Length (Unit: mm)
# Note: In DIFF mode, the Mag selector controls Camera Length.
TEM_DIFF = {
    "MagList": [
        60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800,
        1000, 1200, 1500, 2000
    ],
    "CamList": []
}

# =============================================================================
# 2. STEM MODES
# =============================================================================

# STEM:SM-MAG (Standard STEM Imaging)
# Reference: F200
# MagList: Magnification (Unit: x)
# CamList: Camera Length (Unit: mm)
STEM_SM_MAG = {
    "MagList": [
        10000, 12000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 80000,
        100000, 120000, 150000, 200000, 250000, 300000, 400000, 500000, 600000, 800000,
        1000000, 1200000, 1500000, 2000000, 2500000, 3000000, 4000000, 5000000,
        6000000, 8000000, 10000000, 12000000, 15000000, 20000000, 25000000,
        30000000, 40000000, 50000000, 60000000, 80000000, 100000000, 120000000, 150000000
    ],
    "CamList": [
        15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400,
        500, 600, 800, 1000, 1200, 1500
    ]
}

# STEM:AMAG (Area Mag)
# Reference: PyJEM Offline
# MagList: Magnification (Unit: x)
# CamList: Camera Length (Unit: mm)
STEM_AMAG = {
    "MagList": [
        5000, 6000, 8000, 10000, 12000, 15000, 20000, 25000, 30000, 40000,
        50000, 60000, 80000, 100000, 120000, 150000, 200000, 250000, 300000,
        400000, 500000, 600000, 800000, 1000000, 1200000, 1500000, 2000000
    ],
    "CamList": [
        80, 100, 120, 150, 200, 300, 400, 500, 600, 800,
        1000, 1200, 1500, 2000, 2500
    ]
}

# STEM:SM-LMAG (Low Mag)
# Reference: PyJEM Offline
# MagList: Magnification (Unit: x)
# CamList: Camera Length (Unit: mm)
STEM_SM_LMAG = {
    "MagList": [
        120, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1500,
        2000, 2500, 3000, 4000
    ],
    "CamList": [20]
}

# STEM:ALIGN (Alignment Mode)
# Reference: PyJEM Offline
# MagList: Tilt/Shift Angle (Unit: degrees)
# CamList: Magnification (Unit: x)
STEM_ALIGN = {
    "MagList": [0, 100],
    "CamList": [8000, 10000, 20000]
}

# STEM:ROCKING (Rocking Beam)
# Reference: PyJEM Offline
# MagList: Rocking Angle (Unit: degrees)
# CamList: Magnification (Unit: x)
STEM_ROCKING = {
    "MagList": [10, 15, 20, 25, 30, 40, 50, 60, 80, 100],
    "CamList": [8000, 10000, 20000]
}

# STEM:UUDIFF (Ultra Ultra Diff)
# Reference: PyJEM Offline
# MagList: Magnification (Unit: x)
# CamList: Camera Length (Unit: mm)
STEM_UUDIFF = {
    "MagList": STEM_SM_MAG["MagList"], # Shares range with SM-MAG
    "CamList": [80, 100, 120, 150, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000, 2500]
}

# =============================================================================
# MAIN REGISTRY
# =============================================================================

DEFAULT_TABLES = {
    # TEM
    "TEM:MAG": TEM_MAG,
    "TEM:MAG2": TEM_MAG2,
    "TEM:LOWMAG": TEM_LOWMAG,
    "TEM:SAMAG": TEM_SAMAG,
    "TEM:DIFF": TEM_DIFF,

    # STEM
    "STEM:SM-MAG": STEM_SM_MAG,
    "STEM:AMAG": STEM_AMAG,
    "STEM:SM-LMAG": STEM_SM_LMAG,
    "STEM:ALIGN": STEM_ALIGN,
    "STEM:ROCKING": STEM_ROCKING,
    "STEM:UUDIFF": STEM_UUDIFF
}