# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 21:32:57 2025

@author: Admin
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import subprocess
from PIL import Image, ImageTk
from astropy.io import fits
import gc
import cv2
import RPi.GPIO as GPIO

# Configuration for GPIO-based RF 433MHz Receiver
RECEIVER_PIN = 17  # Change this to the GPIO pin you connected the RF receiver DATA OUT to
GPIO.setmode(GPIO.BCM)
GPIO.setup(RECEIVER_PIN, GPIO.IN)

class FrequencyTunerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Frequency Tuner + SSTV + Auto Tune + Astrometrica")
        self.fs = 44100
        self.signal = None
        self.image_data = None
        self.sstv_image = None
        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.root)
        frame.pack(padx=10, pady=10)

        # Keep all GUI components the same (lowcut/highcut kept for placeholder)
        ttk.Label(frame, text="Low Cut (Hz)").grid(row=0, column=0)
        self.lowcut_entry = ttk.Entry(frame)
        self.lowcut_entry.insert(0, "1000")
        self.lowcut_entry.grid(row=0, column=1)

        ttk.Label(frame, text="High Cut (Hz)").grid(row=1, column=0)
        self.highcut_entry = ttk.Entry(frame)
        self.highcut_entry.insert(0, "3000")
        self.highcut_entry.grid(row=1, column=1)

        ttk.Label(frame, text="Radio Frequency (MHz)").grid(row=2, column=0)
        self.radio_freq_entry = ttk.Entry(frame)
        self.radio_freq_entry.insert(0, "433.0")
        self.radio_freq_entry.grid(row=2, column=1)

        self.radio_selector = ttk.Combobox(frame, state="readonly")
        self.radio_selector['values'] = ["GPIO RF433"]
        self.radio_selector.current(0)
        ttk.Label(frame, text="Select Radio Device").grid(row=3, column=0)
        self.radio_selector.grid(row=3, column=1)

        self.audio_selector = ttk.Combobox(frame, state="readonly")
        self.audio_selector['values'] = ["Not Used"]
        self.audio_selector.current(0)
        ttk.Label(frame, text="Select SSTV Audio Input").grid(row=4, column=0)
        self.audio_selector.grid(row=4, column=1)

        self.auto_start_entry = ttk.Entry(frame)
        self.auto_stop_entry = ttk.Entry(frame)
        self.auto_step_entry = ttk.Entry(frame)
        for i, label in enumerate(["Auto Tune Start (MHz)", "Stop (MHz)", "Step (MHz)"]):
            ttk.Label(frame, text=label).grid(row=5 + i, column=0)
        self.auto_start_entry.insert(0, "433.0")
        self.auto_stop_entry.insert(0, "433.1")
        self.auto_step_entry.insert(0, "0.1")
        self.auto_start_entry.grid(row=5, column=1)
        self.auto_stop_entry.grid(row=6, column=1)
        self.auto_step_entry.grid(row=7, column=1)

        ttk.Button(frame, text="Start Auto Tune", command=self.start_auto_tune).grid(row=8, column=0, pady=5)
        ttk.Button(frame, text="Generate Signal", command=self.generate_signal).grid(row=9, column=0)
        ttk.Button(frame, text="Load WAV File", command=self.load_audio).grid(row=9, column=1)
        ttk.Button(frame, text="Capture from Radio", command=self.capture_radio).grid(row=10, column=0)
        ttk.Button(frame, text="Apply Filter", command=self.process_signal).grid(row=10, column=1)
        ttk.Button(frame, text="Show Spectrum", command=self.plot_spectrum).grid(row=11, column=0)
        ttk.Button(frame, text="View as SSTV", command=self.simulate_sstv).grid(row=11, column=1)
        ttk.Button(frame, text="Export to Astrometrica (FITS)", command=self.export_to_astrometrica).grid(row=13, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="Convert WAV to FITS via Image", command=self.convert_wav_to_fits_via_image).grid(row=14, column=0, columnspan=2, pady=5)

        self.canvas = tk.Canvas(self.root, width=320, height=240, bg='black')
        self.canvas.pack(pady=10)

    def generate_signal(self):
        t = np.linspace(0, 1.0, self.fs)
        self.signal = 0.6 * np.sin(2 * np.pi * 1500 * t) + 0.4 * np.sin(2 * np.pi * 2500 * t)

    def load_audio(self):
        from soundfile import read
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if path:
            self.signal, self.fs = read(path)
            if self.signal.ndim > 1:
                self.signal = self.signal[:, 0]

    def capture_radio(self):
        print("📡 Capturing RF433 signal from GPIO...")
        duration = 5
        transitions = []
        last = GPIO.input(RECEIVER_PIN)
        start_time = time.time()
        while time.time() - start_time < duration:
            val = GPIO.input(RECEIVER_PIN)
            if val != last:
                transitions.append((time.time() - start_time, val))
                last = val
        print(f"✅ Captured {len(transitions)} transitions")
        times, values = zip(*transitions) if transitions else ([0], [0])
        signal = np.array(values, dtype=np.float32)
        self.signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
        self.fs = len(self.signal) / duration

    def process_signal(self):
        if self.signal is None:
            return
        from scipy.signal import butter, lfilter
        lowcut = float(self.lowcut_entry.get())
        highcut = float(self.highcut_entry.get())
        b, a = butter(5, [lowcut / (0.5 * self.fs), highcut / (0.5 * self.fs)], btype='band')
        self.signal = lfilter(b, a, self.signal)

    def plot_spectrum(self):
        if self.signal is None:
            return
        from scipy.signal import spectrogram
        f, t, Sxx = spectrogram(self.signal, self.fs)
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-6), shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title("Spectrogram")
        plt.colorbar(label='dB')
        plt.tight_layout()
        plt.show()

    def simulate_sstv(self):
        if self.signal is None:
            return
        reshaped = np.resize(self.signal, 320 * 240).reshape((240, 320))
        norm = (reshaped - reshaped.min()) / reshaped.ptp() * 255
        img = Image.fromarray(norm.astype(np.uint8)).convert("L")
        img = img.resize((320, 240))
        self.image_data = np.array(img)
        self.sstv_image = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.sstv_image)
        self.canvas.image = self.sstv_image
        gc.collect()

    def start_auto_tune(self):
        print("Auto tune is not supported for GPIO RF433 module.")

    def export_to_astrometrica(self):
        if self.image_data is None:
            print("No image data available for FITS export.")
            return
        fits_file = filedialog.asksaveasfilename(defaultextension=".fits", filetypes=[("FITS files", "*.fits")])
        if not fits_file:
            return
        hdu = fits.PrimaryHDU(self.image_data)
        hdu.writeto(fits_file, overwrite=True)
        print(f"✅ Exported to FITS: {fits_file}")
        try:
            subprocess.Popen(["Astrometrica.exe", fits_file])
        except FileNotFoundError:
            print("Astrometrica.exe not found. Please launch manually.")

    def convert_wav_to_fits_via_image(self):
        wav_file = filedialog.askopenfilename(title="Select SSTV WAV file", filetypes=[("WAV files", "*.wav")])
        if not wav_file:
            return
        messagebox.showinfo("Decode WAV", f"Now decode the WAV file manually using RX-SSTV or Robot36:\n{wav_file}")
        image_file = filedialog.askopenfilename(title="Select Decoded Image", filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if not image_file:
            return
        fits_file = filedialog.asksaveasfilename(defaultextension=".fits", filetypes=[("FITS files", "*.fits")])
        if not fits_file:
            return
        try:
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            if image is None:
                messagebox.showerror("Error", "Could not load image. Try PNG or JPG.")
                return
            hdu = fits.PrimaryHDU(image.astype(np.uint8))
            hdu.writeto(fits_file, overwrite=True)
            messagebox.showinfo("Success", f"FITS saved to: {fits_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Conversion failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FrequencyTunerApp(root)
    root.mainloop()