import matplotlib.pyplot as plt


# ----------------------------
# 7. Visualization
# ----------------------------

def plot_sample_series(X_sample):
    plt.figure(figsize=(10,4))
    for i in range(X_sample.shape[1]):
        plt.plot(X_sample[:,i], label=f'Sensor {i}')
    plt.title("Sample Sensor Time Series")
    plt.legend()
    plt.show()