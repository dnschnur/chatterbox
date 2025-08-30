"""mel-spectrogram extraction in Matcha-TTS"""

import numpy as np
import torch
import torchaudio

mel_basis = {}
hann_window = {}


"""
feat_extractor: !name:matcha.utils.audio.mel_spectrogram
    n_fft: 1920
    num_mels: 80
    sampling_rate: 24000
    hop_size: 480
    win_size: 1920
    fmin: 0
    fmax: 8000
    center: False
"""

def mel_spectrogram(y, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920,
                    fmin=0, fmax=8000, center=False):
    """Copied from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/utils/audio.py
    Set default values according to Cosyvoice's config.
    """

    if isinstance(y, np.ndarray):
        y = torch.tensor(y).float()

    if len(y.shape) == 1:
        y = y[None, ]

    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window  # pylint: disable=global-statement,global-variable-not-assigned
    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = torchaudio.functional.melscale_fbanks(
            n_freqs=int(n_fft // 2 + 1),
            f_min=fmin,
            f_max=fmax,
            n_mels=num_mels,
            sample_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney").T

        mel_basis[str(fmax) + "_" + str(y.device)] = mel.to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    spec = mel_basis[str(fmax) + "_" + str(y.device)] @ spec
    return torch.log(torch.clamp(spec, min=1e-5))
