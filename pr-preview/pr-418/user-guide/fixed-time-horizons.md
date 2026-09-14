# Fixed Time Horizons

A time-to-event prediction must specify when the outcome is evaluated. Use `fixed_time_horizons` to declare those times in the same unit as `times` and the prediction model:

``` python
fixed_time_horizons = [5.0, 10.0]  # years
```

Choose clinically meaningful horizons before inspecting performance. Changing the horizon changes the outcome being validated.


# Update administrative censoring

At a fixed horizon:

- an event observed after the horizon is a 🤨 non-event *at that horizon*;
- event-free follow-up ending before the horizon is 🤬 censored;
- a primary event observed by the horizon remains 🤢; and
- a competing event observed by the horizon remains 💀.


# Explore the horizon

Move the slider to see how the selected horizon changes each observation. The vertical line marks the horizon; information after it is not used.

The symbols are:

- 🤬 censored before the horizon;
- 🤢 primary event;
- 🤨 non-event through the horizon; and
- 💀 competing event.


# Pass horizons to `rtichoke`

``` python
performance_data = rk.prepare_performance_data_times(
    probs=probs,
    reals=reals,
    times=times,
    fixed_time_horizons=[5.0, 10.0],
)
```
