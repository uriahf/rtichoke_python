# Pandas runtime audit

Pandas is no longer declared as a direct runtime dependency.

Remaining source references are legacy cleanup targets:

- `processing/adjustments.py`: `ensure_no_categorical()` is an unused pandas-only helper; the active adjustment pipeline is Polars.
- `processing/send_post_request_to_r_rtichoke.py`: incomplete legacy R-API bridge whose request function is currently a stub.
- `calibration/calibration.py`: only a commented-out pandas import remains.

The dependency benchmark should be rerun after this change to determine whether another runtime dependency still pulls pandas transitively before deleting legacy code.
