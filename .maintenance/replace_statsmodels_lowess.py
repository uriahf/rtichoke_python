from pathlib import Path

calibration_path = Path('src/rtichoke/calibration/calibration.py')
text = calibration_path.read_text()

old_import = 'from polarstate import predict_aj_estimates, prepare_event_table\nfrom ._secondary_cox import calculate_secondary_cox_smooth\n'
new_import = 'from polarstate import predict_aj_estimates, prepare_event_table\nfrom smoothstate import smooth_state_lowess\nfrom ._secondary_cox import calculate_secondary_cox_smooth\n'
if old_import not in text:
    raise SystemExit('Expected import block not found')
text = text.replace(old_import, new_import, 1)

old_local_import = '    from statsmodels.nonparametric.smoothers_lowess import lowess\n\n'
if old_local_import not in text:
    raise SystemExit('Expected statsmodels LOWESS import not found')
text = text.replace(old_local_import, '', 1)

old_block = '''        else:\n            # lowess returns a 2D array where the first column is x and the second is y\n            smoothed = lowess(r, p, it=0)\n            xout = np.linspace(0, 1, 101)\n            yout = np.clip(np.interp(xout, smoothed[:, 0], smoothed[:, 1]), 0.0, 1.0)\n            return pl.DataFrame(\n                {"x": xout, "y": yout, "reference_group": [group_name] * len(xout)}\n            )\n'''
new_block = '''        else:\n            smoothed = smooth_state_lowess(p, r)\n            return smoothed.with_columns(\n                pl.lit(group_name).alias("reference_group")\n            )\n'''
if old_block not in text:
    raise SystemExit('Expected LOWESS implementation block not found')
text = text.replace(old_block, new_block, 1)
calibration_path.write_text(text)

pyproject_path = Path('pyproject.toml')
pyproject = pyproject_path.read_text()
pyproject = pyproject.replace('    "smoothstate>=0.1.0",\n', '    "smoothstate>=0.1.1",\n', 1)
if '    "statsmodels>=0.14.0",\n' not in pyproject:
    raise SystemExit('Expected statsmodels dependency not found')
pyproject = pyproject.replace('    "statsmodels>=0.14.0",\n', '', 1)
pyproject_path.write_text(pyproject)
