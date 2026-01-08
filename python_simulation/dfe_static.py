
import numpy as np
import matplotlib.pyplot as plt
import os

num_bits = 10000

sps = 3
beta = 0.25
span = 10


num_taps = 4
my_snr =8

training_len = 2000
testing_len = 4000

Nf = 9
Nb = 4
delta = 1.0
lam = 0.995
mu_b = 0.001
dd_start_offset = 1000

snr_db_range = np.arange(0, 20, 2)

# Generate random bits
def bit_generation(num_bits):
    return np.random.randint(0, 2, num_bits)

bits = bit_generation(num_bits)

def bit_pairs(bits):
    return bits.reshape(-1, 2)

bit_pairs = bit_pairs(bits)

# QPSK mapping (Gray-coded)
def qpsk_modulation(bit_pairs):
    mapping = {
        (0, 0): (1 + 1j) / np.sqrt(2),
        (0, 1): (-1 + 1j) / np.sqrt(2),
        (1, 1): (-1 - 1j) / np.sqrt(2),
        (1, 0): (1 - 1j) / np.sqrt(2),
    }
    return np.array([mapping[tuple(bp)] for bp in bit_pairs])
qpsk_symbols = qpsk_modulation(bit_pairs)

# Upsampling
def upsample(qpsk_symbols, sps):
    upsampled = np.zeros(len(qpsk_symbols) * sps, dtype=complex)
    upsampled[::sps] = qpsk_symbols
    return upsampled

upsampled = upsample(qpsk_symbols, sps)


# RRC filter design
def rrc_filter(beta, sps, span):
    N = span * sps
    t = np.arange(-N // 2, N // 2 + 1) / sps
    h = np.zeros_like(t)
    for i in range(len(t)):
        if t[i] == 0.0:
            h[i] = 1.0 - beta + (4 * beta / np.pi)
        elif abs(t[i]) == 1 / (4 * beta):
            h[i] = (beta / np.sqrt(2)) * (
                ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))) +
                ((1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
            )
        else:
            numerator = np.sin(np.pi * t[i] * (1 - beta)) + \
                        4 * beta * t[i] * np.cos(np.pi * t[i] * (1 + beta))
            denominator = np.pi * t[i] * (1 - (4 * beta * t[i]) ** 2)
            h[i] = numerator / denominator
    return h / np.sqrt(np.sum(h**2))


rrc = rrc_filter(beta, sps, span)

# Filtered transmit signal

#this is the transmitting signal

tx_signal = np.convolve(upsampled, rrc, mode='same')

# Calculate delay from filter length
delay = (len(rrc) - 1) // 2

# Symbol-rate sampling (to observe clean constellation)
sampled_tx = tx_signal[delay::sps]  # sample_tx is transmitted in the last


# Plot symbol-rate constellation
plt.figure(figsize=(5, 5))
plt.plot(np.real(sampled_tx), np.imag(sampled_tx), 'bo')
plt.title("Sampled RRC Output (Symbol Rate)")
plt.xlabel("In-phase")
plt.ylabel("Quadrature")
plt.grid(True)
plt.axis("equal")
plt.show()

# 	Apply multipath channel	(Linear convolution)
def fading_channel(signal, channel_taps):
    return np.convolve(signal, channel_taps, mode='full')

#  	Matched filter at receiver	(Linear convolution)
def recieved(signal, rrc):
    return np.convolve(signal, rrc , mode='full')

# Timing alignment or channel response	(Cross-correlation)
def correlation(signal1, signal2):
    return np.correlate(signal1, signal2, mode='full')

def adding_noise(faded_signal, snr_db):
    signal_power = np.mean(np.abs(faded_signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(faded_signal)) + 1j*np.random.randn(len(faded_signal)))
    rx_signal = faded_signal + noise
    return rx_signal,noise_power,signal_power

def compute_delay(corr, sps):
    best_delay = np.argmax(corr) - len(upsampled) + 1
    aligned_delay = best_delay + (sps - best_delay % sps) % sps
    return aligned_delay,best_delay

# Multipath fading channel
channel_taps = {
    'CH1': np.array([0.9 + 0.3j, 0.6 - 0.2j, 0.4 + 0.1j, 0.2 - 0.1j]) / np.sqrt(2),

    'CH2': np.array([0.4 + 0.3j, 0.6 - 0.5j, 0.3 + 0.5j, 0.1 - 0.7j]) / np.sqrt(2),

    'CH3': np.array([0.8 + 0.1j, 0.5 + 0.2j, 0.3 - 0.4j, 0.1 + 0.3j]) / np.sqrt(2)
}

for ch_name, taps in channel_taps.items():
    # Channel convolution
    faded_signal= fading_channel(tx_signal, taps)


    # AWGN
    rx_signal,noise_power,signal_power  = adding_noise(faded_signal,my_snr)

    print('Noise power:',noise_power)
    print('Signal power:',signal_power)



    # Matched filtering at receiver
    rx_filtered = recieved(rx_signal, rrc)

    # Estimate best sampling delay using correlation
    corr = correlation(np.abs(rx_filtered), np.abs(upsampled))

    # Sample at symbol rate after delay
    aligned_delay,best_delay= compute_delay(corr, sps)
    print('Best delay:',best_delay)
    print('Aligned delay:',aligned_delay)

    symbols_rx = rx_filtered[aligned_delay::sps]
    symbols_rx = symbols_rx[:len(qpsk_symbols)]
    min_len = min(len(symbols_rx), len(qpsk_symbols))

    # plot recieved signal
    plt.figure()
    plt.plot(np.real(symbols_rx), np.imag(symbols_rx), 'bo', label=ch_name)
    plt.title(f"Received Symbols After Sync ({ch_name}) (Before equalization)")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

# --- QPSK decision and bit mapping ---
def qpsk_decision(symbol):
    real = 1 if symbol.real >= 0 else -1
    imag = 1 if symbol.imag >= 0 else -1
    if real == 1 and imag == 1:
        return (1 + 1j) / np.sqrt(2)     # (0, 0)
    elif real == -1 and imag == 1:
        return (-1 + 1j) / np.sqrt(2)    # (0, 1)
    elif real == -1 and imag == -1:
        return (-1 - 1j) / np.sqrt(2)    # (1, 1)
    else:
        return (1 - 1j) / np.sqrt(2)     # (1, 0)

def qpsk_to_bits(symbol):
    bits = []
    for s in symbol:
        bits.append(0 if s.real >= 0 else 1)
        bits.append(0 if s.imag >= 0 else 1)
    return np.array(bits)

# --- RLS-DFE Equalizer ---
def rls_dfe(rx_symbols, transmitted_symbols, Nf, Nb, delta, lam, mu_b, training_len, dd_start_offset):
    N = len(rx_symbols)
    w_f = np.zeros(Nf, dtype=complex)
    w_b = np.zeros(Nb, dtype=complex)
    P_f = np.eye(Nf, dtype=complex) / delta

    x_f = np.zeros(Nf, dtype=complex)
    x_b = np.zeros(Nb, dtype=complex)
    y_out = np.zeros(N, dtype=complex)
    detected = np.zeros(N, dtype=complex)
    error_mse = np.zeros(N)
    tap_evolution = np.zeros((N,Nf), dtype=complex)
    tap_evolution_b = np.zeros((N,Nb), dtype=complex)

    for n in range(N):
        # Feedforward input
        x_f = np.roll(x_f, 1)
        x_f[0] = rx_symbols[n]

        # Feedback input
        if n == 0:
            x_b[:] = 0
        else:
            x_b = np.roll(x_b, 1)
            x_b[0] = detected[n - 1]

        # Equalizer output
        y = np.dot(w_f.conj(), x_f) - np.dot(w_b.conj(), x_b)
        y_out[n] = y

        # Define reference symbol
        if n < training_len:
            d = transmitted_symbols[n]  # Training mode
        elif n < training_len + dd_start_offset:
            d = transmitted_symbols[n]  # Extended supervised mode (optional)
        else:
            d = qpsk_decision(y)  # Decision-directed mode

        # Error
        e = d - y
        error_mse[n] = np.abs(e)**2

        # RLS update (only feedforward)
        Pi_x = np.dot(P_f, x_f)
        k = Pi_x / (lam + np.dot(x_f.conj().T, Pi_x))
        P_f = (P_f - np.outer(k, Pi_x.conj().T)) / lam
        w_f += k * e.conj()



        # Feedback filter update (LMS style)
        w_b += mu_b * e.conj() * x_b


        # Symbol decision for feedback

        detected[n] = qpsk_decision(y)

        tap_evolution[n] = w_f
        tap_evolution_b[n] =  w_b

        # print("forwar :",tap_evolution[:100])
        # print("forwar :",tap_evolution_b[:100])


    return y_out, detected, error_mse,tap_evolution,tap_evolution_b

def simulate_channel(channel_name, channel_impulse, snr_db, tx_signal, rrc, upsampled, qpsk_symbols,
                     Nf, Nb, delta, lam, mu_b, training_len, dd_start_offset, total_len):
    # Normalize channel
    h = channel_impulse / np.linalg.norm(channel_impulse)
    faded_signal = fading_channel(tx_signal, h)

    # Add noise

    rx_signal,noise_power,signal_power = adding_noise(faded_signal, snr_db)



    # Matched filter
    rx_filtered = recieved(rx_signal, rrc)

    # Symbol timing
    corr = correlation(np.abs(rx_filtered), np.abs(upsampled))

    aligned_delay,best_delay = compute_delay(corr, sps)

    symbols_rx = rx_filtered[aligned_delay::sps][:len(qpsk_symbols)]

    # Equalization
    rx_symbols_used = symbols_rx[:total_len]
    tx_symbols_used = qpsk_symbols[:total_len]

    y_out, detected, error_mse, tap_evolution , tap_evolution_b= rls_dfe(rx_symbols_used, tx_symbols_used, Nf, Nb, delta, lam, mu_b, training_len, dd_start_offset)


    return y_out, detected, error_mse, tap_evolution, tx_symbols_used , tap_evolution_b

def qpsk_to_bits(symbol):
    bits = []
    for s in symbol:
        bits.append(0 if s.real >= 0 else 1)
        bits.append(0 if s.imag >= 0 else 1)
    return np.array(bits)

# --- Helper functions ---

# --- Main function ---
def evaluate_rls_dfe(channel_taps, snr_db_range, tx_signal, rrc, upsampled, qpsk_symbols,
                     Nf, Nb, delta, lam, mu_b, training_len, dd_start_offset):

    total_len = training_len + testing_len
    ber_results_vs_snr = {ch: [] for ch in channel_taps}

    for snr_db in snr_db_range:
        for ch_name, h in channel_taps.items():

            y_out, detected, error_mse, tap_evolution, tx_symbols_used, tap_evolution_b = simulate_channel(
                ch_name, h, snr_db, tx_signal, rrc, upsampled, qpsk_symbols,
                Nf, Nb, delta, lam, mu_b, training_len, dd_start_offset, total_len
            )
            # Plotting (optional)
            if snr_db == my_snr:
                plt.scatter(np.real(y_out), np.imag(y_out), color='blue', alpha=0.4, s=10)
                plt.title(f"Equalized Constellation ({ch_name}, {snr_db} dB)")
                plt.xlabel("In-phase")
                plt.ylabel("Quadrature")
                plt.axis('equal')
                plt.grid(True)
                plt.tight_layout()
                plt.show()

                plt.figure(figsize=(8, 3))
                plt.plot(error_mse)
                plt.title(f"MSE Convergence ({ch_name}, {snr_db} dB)")
                plt.xlabel("Iteration")
                plt.ylabel("MSE")
                plt.grid(True)
                plt.tight_layout()
                plt.show()

                plt.figure(figsize=(10, 5))
                for i in range(min(3, Nf)):
                    plt.plot(np.abs(tap_evolution[:, i]), label=f"Feedforward Tap {i+1}")
                plt.title(f"Feedforward Filter Weights Evolution\n{ch_name} at {snr_db} dB")
                plt.xlabel("Symbol Index")
                plt.ylabel("Tap Magnitude")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

            # BER
            rx_bits = qpsk_to_bits(detected[training_len:])
            tx_bits = qpsk_to_bits(tx_symbols_used[training_len:])
            min_len = min(len(rx_bits), len(tx_bits))
            ber = np.mean(rx_bits[:min_len] != tx_bits[:min_len])
            ber_results_vs_snr[ch_name].append(ber)

    return ber_results_vs_snr



equalized_and_mse_plot = evaluate_rls_dfe(
    channel_taps, snr_db_range,
    tx_signal, rrc, upsampled, qpsk_symbols,
    Nf, Nb, delta, lam, mu_b,
    training_len, dd_start_offset
)

def plot_ber_vs_snr(ber_results_vs_snr, snr_db_range, title="BER vs SNR for Different Channels"):
    plt.figure(figsize=(8, 6))
    for ch, ber_vals in ber_results_vs_snr.items():
        plt.semilogy(snr_db_range, ber_vals, marker='o', label=ch)
    plt.grid(True, which='both')
    plt.title(title)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_ber_vs_snr(equalized_and_mse_plot, snr_db_range)

