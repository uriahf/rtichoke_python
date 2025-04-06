from rtichoke.summary_report.summary_report import render_summary_report


help(render_summary_report)

probs = [0.1, 0.4, 0.8]
reals = [0, 1, 1]
times = [1, 3, 5]
render_summary_report(probs, reals, times)