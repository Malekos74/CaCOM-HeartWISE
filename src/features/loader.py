from pathlib import Path
import librosa


def load_audio(file_path, sr=None):
    """
    Load a wav file.

    Returns:
        signal (np.ndarray), sample_rate (int)
    """
    signal, sr = librosa.load(file_path, sr=sr)
    return signal, sr


def get_all_files(data_dir):
    """
    Returns all wav files in dataset.
    """
    return sorted(Path(data_dir).glob("*.wav"))