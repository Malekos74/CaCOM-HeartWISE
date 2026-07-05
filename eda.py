from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from tqdm import tqdm

DATA_DIR = Path("data/training_data")

patients = []
recordings = []

###########################################################################
# Parse metadata
###########################################################################

txt_files = sorted(DATA_DIR.glob("*.txt"))

for txt in tqdm(txt_files):

    patient = {
        "patient_id": txt.stem
    }

    with open(txt, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    for line in lines:

        if ":" not in line:
            continue

        key, value = line.split(":", 1)

        patient[key.strip()] = value.strip()

    patients.append(patient)

###########################################################################
# Parse recordings
###########################################################################

wav_files = sorted(DATA_DIR.glob("*.wav"))

for wav in tqdm(wav_files):

    info = sf.info(str(wav))

    patient = wav.stem.split("_")[0]

    location = wav.stem.split("_")[1]

    recordings.append({
        "patient_id": patient,
        "location": location,
        "samplerate": info.samplerate,
        "channels": info.channels,
        "duration": info.duration,
        "frames": info.frames
    })

###########################################################################
# DataFrames
###########################################################################

patients = pd.DataFrame(patients)
recordings = pd.DataFrame(recordings)

###########################################################################
# Overview
###########################################################################

print("="*60)
print("PATIENT OVERVIEW")
print("="*60)

print(patients.head())

print("\nPatients:", len(patients))
print("Recordings:", len(recordings))

print("\nColumns")
print(patients.columns)

###########################################################################
# Missing values
###########################################################################

print("\nMissing values")
print(patients.isna().sum())

###########################################################################
# Murmur distribution
###########################################################################

if "Murmur" in patients.columns:

    plt.figure(figsize=(6,4))

    patients["Murmur"].value_counts().plot(kind="bar")

    plt.title("Murmur Distribution")

    plt.tight_layout()

    plt.savefig("plots/eda/murmur_distribution.png")

###########################################################################
# Outcome distribution
###########################################################################

if "Outcome" in patients.columns:

    plt.figure(figsize=(6,4))

    patients["Outcome"].value_counts().plot(kind="bar")

    plt.title("Outcome Distribution")

    plt.tight_layout()

    plt.savefig("plots/eda/outcome_distribution.png")

###########################################################################
# Sex
###########################################################################

if "Sex" in patients.columns:

    plt.figure(figsize=(6,4))

    patients["Sex"].value_counts().plot(kind="bar")

    plt.title("Sex Distribution")

    plt.tight_layout()

    plt.savefig("plots/eda/sex_distribution.png")

###########################################################################
# Age
###########################################################################

if "Age" in patients.columns:

    plt.figure(figsize=(8,4))

    patients["Age"].value_counts().plot(kind="bar")

    plt.title("Age Groups")

    plt.tight_layout()

    plt.savefig("plots/eda/age_groups.png")

###########################################################################
# Recording locations
###########################################################################

plt.figure(figsize=(7,4))

recordings["location"].value_counts().plot(kind="bar")

plt.title("Recording Locations")

plt.tight_layout()

plt.savefig("plots/eda/recording_locations.png")

###########################################################################
# Recording duration
###########################################################################

plt.figure(figsize=(8,4))

plt.hist(recordings["duration"], bins=40)

plt.xlabel("Duration (seconds)")

plt.ylabel("Count")

plt.title("Recording Duration")

plt.tight_layout()

plt.savefig("plots/eda/duration_histogram.png")

###########################################################################
# Number of recordings per patient
###########################################################################

counts = recordings.groupby("patient_id").size()

plt.figure(figsize=(8,4))

plt.hist(counts, bins=15)

plt.xlabel("Recordings")

plt.ylabel("Patients")

plt.title("Recordings per Patient")

plt.tight_layout()

plt.savefig("plots/eda/recordings_per_patient.png")

###########################################################################
# Sampling rates
###########################################################################

print("\nSampling Rates")

print(recordings["samplerate"].value_counts())

###########################################################################
# Duration statistics
###########################################################################

print("\nRecording Duration Statistics")

print(recordings["duration"].describe())

###########################################################################
# Merge metadata
###########################################################################

merged = recordings.merge(
    patients,
    on="patient_id",
    how="left"
)

###########################################################################
# Average recording length by murmur
###########################################################################

if "Murmur" in merged.columns:

    print("\nAverage duration by Murmur")

    print(
        merged.groupby("Murmur")["duration"].describe()
    )

###########################################################################
# Average recording length by outcome
###########################################################################

if "Outcome" in merged.columns:

    print("\nAverage duration by Outcome")

    print(
        merged.groupby("Outcome")["duration"].describe()
    )

###########################################################################
# Print summary
###########################################################################

print("\n")
print("="*60)
print("SUMMARY")
print("="*60)

print(f"Patients             : {len(patients)}")
print(f"Recordings           : {len(recordings)}")
print(f"Mean duration        : {recordings.duration.mean():.2f} s")
print(f"Shortest recording   : {recordings.duration.min():.2f} s")
print(f"Longest recording    : {recordings.duration.max():.2f} s")
print(f"Recording locations  : {recordings.location.unique()}")
print("="*60)