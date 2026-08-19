# Calibration renderer

`calibration_renderer.js` is the isolated D3 implementation used for calibration-parity work against the R `create_summary_report()` reference.

The extraction keeps calibration geometry, histogram behavior, axes, legend, and hover behavior reviewable without mixing changes into discrimination, utility, or performance-table rendering.

The next wiring step is to inject `calibration_renderer_source()` into the generated self-contained HTML in place of the inline `calibration()` implementation. Once wired, hover and visual parity changes should be made only in this renderer.
