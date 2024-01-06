import numpy as np
import math
import matplotlib.pyplot as plt


def p_dot_p(t_a, gamma_a, t_b, gamma_b):
    t_ratio = t_a / t_b
    t_exp = np.where(t_a < t_b, gamma_b, -gamma_a)
    t_max = np.maximum(t_a, t_b)
    num = (gamma_a + 1) * (gamma_b + 1) * t_ratio**t_exp
    den = (gamma_a + gamma_b + 1) * t_max
    return num / den


def solve_weights(t_i, gamma_i, t_r, gamma_r):
    rv = lambda x: np.float64(x).reshape(-1, 1)
    cv = lambda x: np.float64(x).reshape(1, -1)
    A = p_dot_p(rv(t_i), rv(gamma_i), cv(t_i), cv(gamma_i))
    B = p_dot_p(rv(t_i), rv(gamma_i), cv(t_r), cv(gamma_r))
    X = np.linalg.solve(A, B)
    return X


def power_ema(y, gamma, t):
    ema_y = np.zeros_like(y)
    ema_y[0] = y[0]
    for i in range(1, len(y)):
        beta_t = (1 - 1 / t[i]) ** (gamma + 1)
        ema_y[i] = beta_t * ema_y[i - 1] + (1 - beta_t) * y[i]
    return ema_y


if __name__ == "__main__":
    N = 1000
    t = np.arange(1, N + 1)

    checkpoint_freq = 100
    checkpoint_index = np.arange(checkpoint_freq - 1, N, checkpoint_freq)
    # [99, 199, 299, 399, 499, 599, 699, 799, 899, 999]

    print(checkpoint_index)

    y_t_1d = (
        10 + math.sqrt(N) * np.sin(t / 300) + np.cumsum(np.random.normal(0, 0.5, N))
    )

    gamma_1 = 3
    gamma_2 = 16
    gamma_3 = 8

    y_t_ema1 = power_ema(y_t_1d, gamma_1, t)
    y_t_ema2 = power_ema(y_t_1d, gamma_2, t)

    y_t_ema3_ground_truth = power_ema(y_t_1d, gamma_3, t)

    ema3_last_ground_truth = y_t_ema3_ground_truth[-1]
    last_index = t[-1]

    t_checkpoint = t[checkpoint_index]
    
    ts = np.concatenate((t_checkpoint, t_checkpoint))
    gammas = np.concatenate(
        (
            np.ones_like(checkpoint_index) * gamma_1,
            np.ones_like(checkpoint_index) * gamma_2,
        )
    )

    x = solve_weights(ts, gammas, last_index, gamma_3)
    emapoints = np.concatenate((y_t_ema1[checkpoint_index], y_t_ema2[checkpoint_index]))
    print(x)

    ema3_last_approximated = np.dot(x.reshape(-1), emapoints.reshape(-1))

    print(f"EMA3 Last Ground Truth: {ema3_last_ground_truth}")
    print(f"EMA3 Last Approximated: {ema3_last_approximated}")

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.plot(t, y_t_1d, label="Original Data", color="gray", alpha=0.7)
    plt.plot(t, y_t_ema1, label=f"EMA Gamma={gamma_1}", color="blue")
    plt.plot(t, y_t_ema2, label=f"EMA Gamma={gamma_2}", color="green")
    plt.plot(
        t,
        y_t_ema3_ground_truth,
        label=f"EMA Gamma={gamma_3} (Ground Truth)",
        color="red",
    )

    plt.scatter(
        t[checkpoint_index],
        y_t_1d[checkpoint_index],
        color="black",
        marker="x",
        label="Checkpoints",
    )

    plt.scatter(
        last_index,
        ema3_last_ground_truth,
        color="red",
        marker="x",
        label="EMA3 Last Ground Truth",
    )
    plt.scatter(
        last_index,
        ema3_last_approximated,
        color="orange",
        marker="x",
        label="EMA3 Last Approximated",
    )

    plt.title("Power Exponential Moving Average (EMA) Comparison per Gamma and its approximation")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.show()

    save_path = "ema_eq.png"
    plt.savefig(save_path, dpi=300)
