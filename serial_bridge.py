"""
serial_bridge.py — AD8232 → Streamlit Live Bridge
====================================================
Reads raw ECG data from AD8232 via USB serial and stores
it in a shared buffer that app.py reads.

Run this LOCALLY (not on Streamlit Cloud) alongside the app.

Usage:
    python serial_bridge.py --port COM3 --baud 115200   # Windows
    python serial_bridge.py --port /dev/ttyUSB0         # Linux
    python serial_bridge.py --port /dev/cu.usbmodem...  # macOS

AD8232 Arduino sketch should output one integer per line:
    Serial.println(analogRead(A0));

The bridge writes a rolling 10-second buffer to:
    /tmp/cardiosense_live.npy
Which app.py reads and displays.
"""

import argparse
import time
import numpy as np
import serial
import serial.tools.list_ports
from pathlib import Path
import signal
import sys

BUFFER_SECONDS = 30
SAMPLE_RATE    = 250
BUFFER_SIZE    = BUFFER_SECONDS * SAMPLE_RATE
BUFFER_FILE    = Path("/tmp/cardiosense_live.npy")
META_FILE      = Path("/tmp/cardiosense_meta.txt")

buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
ptr    = 0
running = True


def signal_handler(sig, frame):
    global running
    print("\n⏹ Stopping serial bridge...")
    running = False
    sys.exit(0)


def list_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found.")
        return
    print("Available serial ports:")
    for p in ports:
        print(f"  {p.device}  —  {p.description}")


def normalize_adc(val: float, bits: int = 10) -> float:
    """Normalize Arduino ADC (0–1023) to float centered at 0."""
    max_val = (2 ** bits) - 1
    return (val / (max_val / 2)) - 1.0


def run_bridge(port: str, baud: int = 115200, adc_bits: int = 10):
    global ptr

    print(f"🔌 Connecting to {port} @ {baud} baud...")
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # wait for Arduino reset
        ser.reset_input_buffer()
        print(f"✓ Connected! Reading ECG data...")
        print(f"  Buffer: {BUFFER_SECONDS}s ({BUFFER_SIZE} samples)")
        print(f"  Output: {BUFFER_FILE}")
        print(f"  Press Ctrl+C to stop.\n")
    except serial.SerialException as e:
        print(f"✗ Failed to connect: {e}")
        print("  Run with --list to see available ports")
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)

    sample_count = 0
    last_save = time.time()
    last_status = time.time()

    while running:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            val = float(line)
            normalized = normalize_adc(val, adc_bits)

            # Roll buffer
            buffer[ptr] = normalized
            ptr = (ptr + 1) % BUFFER_SIZE
            sample_count += 1

            # Save buffer every 200ms
            now = time.time()
            if now - last_save >= 0.2:
                # Save as ordered array (not ring buffer order)
                ordered = np.roll(buffer, -ptr)
                np.save(BUFFER_FILE, ordered)
                with open(META_FILE, 'w') as f:
                    f.write(f"{SAMPLE_RATE},{sample_count},{time.time()}")
                last_save = now

            # Status every 5s
            if now - last_status >= 5.0:
                hr_est = 0
                if sample_count > SAMPLE_RATE * 3:
                    # Rough HR from buffer
                    recent = buffer[max(0, ptr - SAMPLE_RATE * 10):ptr]
                    if len(recent) > 10:
                        from scipy.signal import find_peaks
                        diff = np.diff(recent, prepend=recent[0]) ** 2
                        win = int(0.15 * SAMPLE_RATE)
                        intg = np.convolve(diff, np.ones(win) / win, mode='same')
                        thr = np.mean(intg) + 0.5 * np.std(intg)
                        peaks, _ = find_peaks(intg, height=thr,
                                              distance=int(0.25 * SAMPLE_RATE))
                        if len(peaks) > 2:
                            rr = np.diff(peaks) / SAMPLE_RATE * 1000
                            hr_est = round(60000 / np.mean(rr))
                print(f"  ✓ {sample_count:,} samples  |  HR ≈ {hr_est} bpm  |  "
                      f"ADC last: {val:.0f}  normalized: {normalized:.3f}")
                last_status = now

        except ValueError:
            pass   # skip malformed lines
        except Exception as e:
            print(f"Read error: {e}")
            time.sleep(0.1)

    ser.close()
    print("Connection closed.")


def main():
    parser = argparse.ArgumentParser(description="AD8232 → CardioSense serial bridge")
    parser.add_argument("--port",  default=None, help="Serial port (e.g. COM3, /dev/ttyUSB0)")
    parser.add_argument("--baud",  type=int, default=115200, help="Baud rate (default 115200)")
    parser.add_argument("--bits",  type=int, default=10, help="ADC resolution bits (default 10 for Arduino)")
    parser.add_argument("--list",  action="store_true", help="List available serial ports and exit")
    args = parser.parse_args()

    if args.list:
        list_ports()
        return

    if not args.port:
        print("✗ No port specified. Run with --list to see available ports.")
        print("  Example: python serial_bridge.py --port COM3")
        list_ports()
        sys.exit(1)

    run_bridge(args.port, args.baud, args.bits)


if __name__ == "__main__":
    main()