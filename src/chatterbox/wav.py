"""Utilities for working with PCM WAV files and audio."""

import wave

from pathlib import Path

import numpy as np
import torch

from torch import Tensor

from .models.utils import Audio


def read_wav(path: Path, device: torch.device) -> Audio:
  """Reads a .wav file and returns it as a mono (1D) float32 tensor along with its sample rate."""
  with wave.open(str(path), 'rb') as wav:
    num_channels = wav.getnchannels()
    sample_width = wav.getsampwidth()
    sample_rate = wav.getframerate()
    # Copy into a bytearray because PyTorch tensors need to be backed by a writeable buffer.
    frames = bytearray(wav.readframes(wav.getnframes()))

  match sample_width:
    case 1:
      wav_dtype = torch.uint8
    case 2:
      wav_dtype = torch.int16
    case _:
      wav_dtype = torch.int32

  audio = torch.frombuffer(frames, dtype=wav_dtype).to(device=device, dtype=torch.float32)
  # Separate the interleaved L/R channel values, reshape into two sequences, and merge to mono.
  audio = torch.mean(audio.view(-1, num_channels).permute(1, 0), dim=0)

  # Scale the values to match the original sample width
  if wav_dtype == torch.uint8:
    audio = (audio - 128.0) / 128.0
  else:
    audio = audio / torch.iinfo(wav_dtype).max

  return Audio(audio, sample_rate)


def write_wav(path: Path, audio: Tensor, sample_rate: int):
  """Writes the given audio data to the given path as a 16-bit PCM WAV file."""
  # If the audio is stereo, interleave the left and right samples.
  if audio.dim() == 2:
    audio = audio.permute(1, 0).flatten()

  # Normalize and scale to the correct range before converting to int16.
  audio = torch.clamp(audio, min=-1.0, max=1.0) * 32767.0
  audio = audio.to(dtype=torch.int16)

  with wave.open(str(path), 'wb') as wav:
    wav.setframerate(sample_rate)
    wav.setnchannels(audio.dim())
    wav.setsampwidth(2)
    wav.writeframes(audio.numpy().tobytes())
