import numpy as np
import matplotlib.pyplot as plt
import random


def generate_synthetic_timeseries(length=3000, num_incidents=10):
    t = np.arange(length)
    base_signal = np.sin(2 * np.pi * t / 100) * 10
    noise = np.random.normal(0, 1.0, length)

    series = base_signal + noise
    labels = np.zeros(length, dtype=int)

    incident_types = ["spike", "variance_drop"]
    available_indices = list(range(150, length - 150))

    for _ in range(num_incidents):
        if not available_indices:
            break

        start = random.choice(available_indices)
        duration = random.randint(30, 90)
        end = min(start + duration, length - 1)
        inc_type = random.choice(incident_types)
        labels[start:end] = 1

        precursor_len = 20

        if inc_type == "spike":
            # simulate increasing vibrations
            for i in range(precursor_len):
                idx = start - precursor_len + i
                series[idx] += np.random.normal(0, 1.5 * (i / precursor_len)) + (i / precursor_len) * 2

            spike_power = random.uniform(10, 20)
            series[start:end] += spike_power + np.random.normal(0, 2, end - start)

        elif inc_type == "variance_drop":
            # heavy rattling before sensor freeze
            for i in range(precursor_len):
                idx = start - precursor_len + i
                series[idx] += np.random.normal(0, 3 * (i / precursor_len))

            series[start:end] = np.random.normal(0, 0.2, end - start)

        available_indices = [idx for idx in available_indices if idx < start - 40 or idx > end + 40]

    return t, series, labels


if __name__ == "__main__":
    t, series, labels = generate_synthetic_timeseries(length=3000, num_incidents=15)

    plt.figure(figsize=(15, 5))
    plt.plot(t, series, label='Time Series (Dynamic)', color='blue', alpha=0.7)
    plt.fill_between(t, series.min() - 5, series.max() + 5,
                     where=(labels == 1), color='red', alpha=0.3, label='Incident (Anomaly)')

    plt.title('Dynamically Generated Time Series with Anomalies')
    plt.xlabel('Time Steps')
    plt.ylabel('Sensor Value')
    plt.legend()
    plt.tight_layout()
    plt.show()