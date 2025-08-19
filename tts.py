"""Chatterbox TTS optimized for Apple Silicon on low-memory devices."""

import argparse
import contextlib
import random
import os

from functools import cached_property
from pathlib import Path

import numpy as np
import torch
import torchaudio

from chatterbox.tts import ChatterboxTTS


DEVICE = torch.device('mps')
DTYPE = torch.float32

DEFAULT_EXAGGERATION = 0.3
DEFAULT_TEMPERATURE = 0.5


class StableSeed(contextlib.AbstractContextManager):
  """Context manager that resets all random seeds at the start of the context block."""

  def __init__(self, seed: int):
    self.seed = seed

  def __enter__(self):
    torch.manual_seed(self.seed)
    random.seed(self.seed)
    np.random.seed(self.seed)
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    return False


class TTS:
  """Chatterbox TTS wrapper that initializes it and generates speech.

  Usage:

    tts = TTS(seed=42)
    tts.initialize(voice='voice.wav')
    tts.generate('Test', output='output.wav')
  """

  def __init__(self, seed: int | None = None):
    self.seed = StableSeed(seed) if seed else contextlib.nullcontext()

  @cached_property
  def model(self) -> ChatterboxTTS:
    """Reference to the ChatterboxTTS model."""
    return ChatterboxTTS.from_pretrained(device='mps')

  def initialize(self, voice: Path | None = None):
    """Initializes Chatterbox, optionally using the given voice."""
    if voice:
      with self.seed:
        self.model.prepare_conditionals(voice)

  def generate(
      self,
      text: str,
      output_path: Path,
      exaggeration: float = DEFAULT_EXAGGERATION,
      temperature: float = DEFAULT_TEMPERATURE):
    """Generates the given text, with the given paramters, to the given output path."""
    with self.seed:
      wav = self.model.generate(
          text, exaggeration=exaggeration, cfg_weight=0.5, temperature=temperature)
    torchaudio.save(output_path, wav, self.model.sr)


def main(args: argparse.Namespace):
  """Initializes Chatterbox with the given voice, if any, and generates the given text."""
  tts = TTS(seed=args.seed)
  tts.initialize(voice=args.voice)
  tts.generate(
      args.text, args.output, exaggeration=args.exaggeration, temperature=args.temperature)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='generate')

  parser.add_argument('text')

  parser.add_argument('-e', '--exaggeration', type=float, default=0.3)
  parser.add_argument('-t', '--temperature', type=float, default=0.5)

  parser.add_argument(
      '-o', '--output', type=Path, default='output.wav',
      help='Path to a .wav file in which to place the generated speech.')
  parser.add_argument(
      '-v', '--voice', type=Path,
      help='Path to a .wav file containing a reference voice to clone.')

  parser.add_argument(
      '-s', '--seed', type=int,
      help='Manual seed for consistency between generations. If not given, outputs are unstable.')

  main(parser.parse_args())
