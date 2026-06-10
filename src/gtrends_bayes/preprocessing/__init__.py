"""Preprocessing pipeline implementing OECD Annex A (Woloszko 2020).

Order matters:
    1. average_samples       (multi_sample.py)
    2. log + remove_long_term_drift   (bias_removal.py)
    3. yoy_log_diff          (seasonality.py)
    4. correct_jan_breaks    (breaks.py)
"""
