import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats
from numpy.typing import NDArray
import config


np.random.seed(config.SEED)
random.seed(config.SEED)


def _inject_traffic_spike(series: NDArray[np.float64], start: int, end: int, precursor_len: int) -> None:
    """Injects a sudden burst with a weak, noisy precursor ramp."""
    for i in range(precursor_len):
        idx = start - precursor_len + i
        series[idx] += np.random.normal(0, 2.5 * (i / precursor_len))
    series[start:end] += random.uniform(15, 30) + np.random.normal(0, 4, end - start)


def _inject_resource_saturation(series: NDArray[np.float64], start: int, end: int, precursor_len: int) -> None:
    """Injects a gradual ramp toward a capacity ceiling with noisy approach."""
    for i in range(precursor_len):
        idx = start - precursor_len + i
        series[idx] += (i / precursor_len) * 8 + np.random.normal(0, 1.5)
    series[start:end] = np.clip(series[start:end] + 25 + np.random.normal(0, 3, end - start), a_min=None, a_max=config.MAX_CAPACITY)


def _inject_memory_leak(series: NDArray[np.float64], start: int, end: int) -> None:
    """Injects a slow, subtle upward drift without an obvious precursor."""
    leak_magnitude = random.uniform(0.15, 0.35)
    leak_len = end - start
    drift = np.cumsum(np.random.normal(leak_magnitude, 0.08, leak_len))
    series[start:end] += drift


def _inject_cascading_failure(series: NDArray[np.float64], start: int, end: int, precursor_len: int) -> None:
    """Injects a brief dip followed by an explosive spike to simulate cascade propagation."""
    dip_len = min(precursor_len, (end - start) // 3)
    for i in range(dip_len):
        idx = start - dip_len + i
        if 0 <= idx < len(series):
            series[idx] -= random.uniform(5, 12)
    spike_phase = end - start
    series[start:end] += np.linspace(0, random.uniform(20, 40), spike_phase) + np.random.normal(0, 3, spike_phase)


def _inject_false_precursors(series: NDArray[np.float64], labels: NDArray[np.int_], num_decoys: int, length: int) -> None:
    """Injects noise bursts that mimic pre-incident patterns but are not followed by any incident."""
    available = [i for i in range(config.EDGE_BUFFER, length - config.EDGE_BUFFER) if labels[i] == 0]
    for _ in range(num_decoys):
        if len(available) < config.PRECURSOR_LEN * 2:
            break
        center = random.choice(available)
        safe_zone = labels[center - config.PRECURSOR_LEN: center + config.PRECURSOR_LEN]
        if np.sum(safe_zone) > 0:
            continue
        for i in range(config.PRECURSOR_LEN):
            idx = center - config.PRECURSOR_LEN + i
            if 0 <= idx < length:
                series[idx] += np.random.normal(0, 2.5 * (i / config.PRECURSOR_LEN))
        available = [idx for idx in available if abs(idx - center) > config.INCIDENT_BUFFER]


def generate_synthetic_timeseries(
    length: int = config.SERIES_LENGTH,
    num_incidents: int = config.NUM_INCIDENTS,
    num_false_precursors: int = config.NUM_FALSE_PRECURSORS,
) -> tuple[NDArray[np.int_], NDArray[np.float64], NDArray[np.int_], list[dict]]:
    """Generates a synthetic cloud telemetry series with heavy-tailed noise, baseline drift, 4 incident types, and false precursors."""
    t = np.arange(length)
    base_signal = np.sin(2 * np.pi * t / config.DIURNAL_PERIOD) * 10 + 50

    heavy_tail_noise = stats.t.rvs(df=config.HEAVY_TAIL_DF, scale=config.NOISE_STD, size=length, random_state=config.SEED)
    drift = np.cumsum(np.random.normal(0, config.DRIFT_STD, length))

    series = base_signal + heavy_tail_noise + drift
    labels = np.zeros(length, dtype=int)

    available_indices = list(range(config.EDGE_BUFFER, length - config.EDGE_BUFFER))
    incident_metadata = []

    for _ in range(num_incidents):
        if not available_indices:
            break

        start = random.choice(available_indices)
        duration = random.randint(config.MIN_INC_DURATION, config.MAX_INC_DURATION)
        end = min(start + duration, length - 1)
        inc_type = random.choice(config.INCIDENT_TYPES)
        labels[start:end] = 1

        if inc_type == "traffic_spike":
            _inject_traffic_spike(series, start, end, config.PRECURSOR_LEN)
        elif inc_type == "resource_saturation":
            _inject_resource_saturation(series, start, end, config.PRECURSOR_LEN)
        elif inc_type == "memory_leak":
            _inject_memory_leak(series, start, end)
        elif inc_type == "cascading_failure":
            _inject_cascading_failure(series, start, end, config.PRECURSOR_LEN)

        incident_metadata.append({"type": inc_type, "start": start, "end": end})
        available_indices = [idx for idx in available_indices if idx < start - config.INCIDENT_BUFFER or idx > end + config.INCIDENT_BUFFER]

    _inject_false_precursors(series, labels, num_false_precursors, length)

    return t, series, labels, incident_metadata


if __name__ == "__main__":
    t, series, labels, metadata = generate_synthetic_timeseries(length=3000, num_incidents=15)

    limit = int(len(series) * 0.30)
    series = series[:limit]
    t = t[:limit]
    labels = labels[:limit]
    plt.figure(figsize=(15, 5))

    plt.plot(t, series, label='Time Series', color='blue', alpha=0.7)
    plt.fill_between(t, series.min() - 5, series.max() + 5, where=(labels == 1), color='red', alpha=0.3, label='Incident')

    plt.title('Synthetic Telemetry with Heavy-Tailed Noise and Drift')
    plt.xlabel('Time Steps')
    plt.ylabel('Sensor Value')
    plt.legend()
    plt.tight_layout()
    plt.show()