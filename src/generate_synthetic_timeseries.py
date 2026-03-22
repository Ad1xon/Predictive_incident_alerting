import numpy as np
import matplotlib.pyplot as plt
import random

from numpy.typing import NDArray
import config


np.random.seed(config.SEED)
random.seed(config.SEED)

def generate_synthetic_timeseries(length: int = config.SERIES_LENGTH, num_incidents: int = config.NUM_INCIDENTS) -> tuple[NDArray[np.int_], NDArray[np.float64], NDArray[np.int_]]:
    """Generates a synthetic cloud telemetry time series with injected traffic spikes and resource saturation incidents."""
    t = np.arange(length)
    base_signal = np.sin(2 * np.pi * t / config.DIURNAL_PERIOD) * 10 + 50
    noise = np.random.normal(0, 2.0, length)

    series = base_signal + noise
    labels = np.zeros(length, dtype=int)

    incident_types = ["traffic_spike", "resource_saturation"]
    available_indices = list(range(config.EDGE_BUFFER, length - config.EDGE_BUFFER))

    for _ in range(num_incidents):
        if not available_indices:
            break

        start = random.choice(available_indices)
        duration = random.randint(config.MIN_INC_DURATION, config.MAX_INC_DURATION)
        end = min(start + duration, length - 1)
        inc_type = random.choice(incident_types)
        labels[start:end] = 1

        precursor_len = config.PRECURSOR_LEN

        if inc_type == "traffic_spike":
            for i in range(precursor_len):
                idx = start - precursor_len + i
                series[idx] += np.random.normal(0, 5 * (i / precursor_len))
            series[start:end] += random.uniform(30, 50) + np.random.normal(0, 5, end - start)

        elif inc_type == "resource_saturation":
            for i in range(precursor_len):
                idx = start - precursor_len + i
                series[idx] += (i / precursor_len) * 15
            series[start:end] = np.clip(series[start:end] + 40, a_min=None, a_max=config.MAX_CAPACITY)

        available_indices = [idx for idx in available_indices if idx < start - config.INCIDENT_BUFFER or idx > end + config.INCIDENT_BUFFER]

    return t, series, labels


if __name__ == "__main__":
    t, series, labels = generate_synthetic_timeseries(length=3000, num_incidents=15)

    plt.figure(figsize=(15, 5))
    plt.plot(t, series, label='Time Series (Dynamic)', color='blue', alpha=0.7)
    plt.fill_between(t, series.min() - 5, series.max() + 5, where=(labels == 1), color='red', alpha=0.3, label='Incident (Anomaly)')

    plt.title('Dynamically Generated Time Series with Anomalies')
    plt.xlabel('Time Steps')
    plt.ylabel('Sensor Value')
    plt.legend()
    plt.tight_layout()
    plt.show()