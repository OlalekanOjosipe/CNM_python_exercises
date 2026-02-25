# lesson4_functions.py
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Exercise 4.1 - Password generation + compliance check
# ------------------------------------------------------------
def Password():
    a = np.random.randint(10, 13)  # 10, 11, 12
    b = np.random.randint(33, 123, a)  # 33..122 inclusive
    password = ""
    for i in range(a):
        password += chr(b[i])
    return password


def Check():
    tries = 0
    while True:
        tries += 1
        pw = Password()

        low = 0
        high = 0
        digit = 0
        special = 0

        for j in range(len(pw)):
            o = ord(pw[j])

            # digits
            if 48 <= o <= 57:
                digit += 1
            # uppercase
            elif 65 <= o <= 90:
                high += 1
            # lowercase
            elif 97 <= o <= 122:
                low += 1
            # everything else inside ASCII 33..122 is "special"
            else:
                special += 1

        if (low >= 2) and (high >= 2) and (digit >= 2) and (special >= 2):
            return pw, tries


# ------------------------------------------------------------
# Exercise 4.2 - Caesar cipher (uppercase letters only)
# ------------------------------------------------------------
def Caesar(message, m=8, mode="encrypt"):
    msg = message.upper()
    if mode == "decrypt":
        m = -m
    out = ""
    for ch in msg:
        o = ord(ch)
        if 65 <= o <= 90:  # 'A'..'Z'
            shifted = (o - 65 + m) % 26 + 65
            out += chr(shifted)
        else:
            out += ch
    return out

# ------------------------------------------------------------
# Exercise 4.3 - Standardize / transform data
# ------------------------------------------------------------
def standardize(data, mean=0, sdev=1):
    data = np.asarray(data)
    mu = np.mean(data)
    sigma = np.std(data)
    if sigma == 0:
        # Avoid division by zero if data is constant
        return np.full_like(data, mean, dtype=float)
    z = (data - mu) / sigma
    out = z * sdev + mean
    return out


# ------------------------------------------------------------
# Exercise 4.4 - Temperature plots for provided files
# ------------------------------------------------------------
def TempPlot(filenames, path=""):
    results = []

    for i in range(len(filenames)):
        fname = filenames[i]
        pathname = path + fname

        metadata = np.loadtxt(fname=pathname, delimiter=",", skiprows=1)
        year = metadata[:, 0]
        data = metadata[:, 1:].copy()  # shape (nyears, 12)

        months = np.arange(1, 13)

        # Monthly stats (across years)
        mean_monthly = np.mean(data, axis=0)
        med_monthly = np.median(data, axis=0)
        std_monthly = np.std(data, axis=0)

        # Annual stats (across months)
        mean_annual = np.mean(data, axis=1)
        med_annual = np.median(data, axis=1)
        std_annual = np.std(data, axis=1)

        # Store results
        results.append({
            "filename": fname,
            "year": year,
            "months": months,
            "mean_monthly": mean_monthly,
            "med_monthly": med_monthly,
            "std_monthly": std_monthly,
            "mean_annual": mean_annual,
            "med_annual": med_annual,
            "std_annual": std_annual
        })

        # Plot
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))
        fig.tight_layout(pad=3.0)

        ax[0, 0].plot(months, mean_monthly, marker="o", linestyle="none")
        ax[0, 0].plot(months, med_monthly, marker="o", linestyle="none")
        ax[0, 0].set_title("Monthly Mean & Median")
        ax[0, 0].set_xlabel("Month")
        ax[0, 0].set_ylabel("Temp")

        ax[0, 1].plot(months, std_monthly, marker="o", linestyle="none")
        ax[0, 1].set_title("Monthly Standard Deviation")
        ax[0, 1].set_xlabel("Month")
        ax[0, 1].set_ylabel("Temp")

        ax[1, 0].plot(year, mean_annual)
        ax[1, 0].plot(year, med_annual)
        ax[1, 0].set_title("Annual Mean & Median")
        ax[1, 0].set_xlabel("Year")
        ax[1, 0].set_ylabel("Temp")

        ax[1, 1].plot(year, std_annual)
        ax[1, 1].set_title("Annual Standard Deviation")
        ax[1, 1].set_xlabel("Year")
        ax[1, 1].set_ylabel("Temp")

        plt.suptitle(fname, y=1.02, fontsize=16)
        plt.show()

    return results


# ------------------------------------------------------------
# Exercise 4.5 - Cubic root iteration
# ------------------------------------------------------------
def cubicroot(a, Tol=1e-8):
    if a == 0:
        return 0.0, 0

    x0 = a / 3.0
    diff = 1.0
    steps = 0

    while diff >= Tol:
        x1 = (2.0 / 3.0) * x0 + a / (3.0 * (x0 ** 2))
        diff = abs(x1 - x0)
        x0 = x1
        steps += 1

        if steps > 10_000:
            break

    return x0, steps


# ------------------------------------------------------------
# Exercise 4.6 - Sieve of Eratosthenes
# ------------------------------------------------------------
def Eras(n):
    if n < 2:
        return np.array([], dtype=int)

    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0:2] = False

    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i * i: n + 1: i] = False

    return np.where(is_prime)[0]

