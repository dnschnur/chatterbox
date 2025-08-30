from functools import lru_cache

import torch
import torchaudio

from torch import Tensor


@lru_cache()
def mel_basis(hp):
  assert hp.fmax <= hp.sample_rate // 2
  return torchaudio.functional.melscale_fbanks(
      n_freqs=int(hp.n_fft // 2 + 1),
      f_min=hp.fmin,
      f_max=hp.fmax,
      n_mels=hp.num_mels,
      sample_rate=hp.sample_rate,
      norm='slaney',
      mel_scale='slaney').T


def preemphasis(wav: Tensor, hp) -> Tensor:
  assert hp.preemphasis != 0
  wav = torchaudio.functional.lfilter([1, -hp.preemphasis], [1], wav)
  return torch.clip(wav, -1, 1)


def melspectrogram(wav: Tensor, hp, pad=True) -> Tensor:
  if hp.preemphasis > 0:
    wav = preemphasis(wav, hp)
    assert torch.abs(wav).max() - 1 < 1e-07

  spec_complex = torch.stft(
      wav,
      n_fft=hp.n_fft,
      hop_length=hp.hop_size,
      win_length=hp.win_size,
      window=torch.hann_window(hp.win_size, device=wav.device),
      center=pad,
      pad_mode='reflect',
      normalized=False,
      return_complex=True)

  spec_magnitudes = torch.abs(spec_complex)

  if hp.mel_power != 1.0:
    spec_magnitudes **= hp.mel_power

  # Get the mel and convert magnitudes->db
  mel = mel_basis(hp).to(spec_magnitudes.device) @ spec_magnitudes
  if hp.mel_type == 'db':
    mel = _amp_to_db(mel, hp)

  # Normalise the mel from db to 0,1
  if hp.normalized_mels:
    mel = _normalize(mel, hp).to(dtype=torch.float32)

  assert not pad or mel.shape[1] == 1 + len(wav) // hp.hop_size   # Sanity check
  return mel   # (M, T)


def _amp_to_db(mel, hp):
  return 20 * torch.log10(torch.maximum(hp.stft_magnitude_min, mel))


def _normalize(mel, hp, headroom_db=15):
  min_level_db = 20 * torch.log10(hp.stft_magnitude_min)
  return (mel - min_level_db) / (-min_level_db + headroom_db)
