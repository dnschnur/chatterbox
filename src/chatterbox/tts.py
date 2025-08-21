"""Chatterbox TTS optimized for Apple Silicon on low-memory devices."""

from __future__ import annotations

import argparse
import contextlib
import random
import os

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional
import torchaudio

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond

# Hugging Face repository ID for pre-trained model weights.
HF_REPO_ID = 'ResembleAI/chatterbox'

PRETRAINED_FILES = [
    've.safetensors', 't3_cfg.safetensors', 's3gen.safetensors', 'tokenizer.json', 'conds.pt']

# System constants
DEVICE = torch.device('mps')
DTYPE = torch.bfloat16

# Punctuation that's output by LLMs or uncommon in the dataset.
UNCOMMON_PUNCTUATION = str.maketrans({
  '…': ', ',
  ':': ',',
  ';': ', ',
  '—': '-',
  '–': '-',
  '“': '"',
  '”': '"',
  '‘': '\'',
  '’': '\'',
})


def clean_text(text: str) -> str:
  """Cleans up one or more chunks of input text."""
  if not text:
    return text

  # Replace punctuation from LLMs or containing characters not often seen in the dataset.
  text = text.replace('...', ', ').replace(' ,', ', ').translate(UNCOMMON_PUNCTUATION)

  # Capitalize and remove extra spaces
  text = ' '.join(text.rstrip(' ').capitalize().split())

  # Append a period if there's no punctuation at the end of the input.
  return text if text.endswith(('.', '!', '?', '-', ',')) else text + '.'


@dataclass
class Conditionals:
  """Conditionals for T3 and S3Gen

  - T3 conditionals:
    - speaker_emb
    - clap_emb
    - cond_prompt_speech_tokens
    - cond_prompt_speech_emb
    - emotion_adv

  - S3Gen conditionals:
    - prompt_token
    - prompt_token_len
    - prompt_feat
    - prompt_feat_len
    - embedding
  """
  t3: T3Cond
  gen: dict

  def to(self, device: torch.device):
    self.t3 = self.t3.to(device=device)
    for k, v in self.gen.items():
      if torch.is_tensor(v):
        self.gen[k] = v.to(device=device)
    return self

  @classmethod
  def load(cls, fpath, map_location='cpu'):
    # Always load to CPU first to handle CUDA-saved models.
    kwargs = torch.load(fpath, map_location=torch.device('cpu'), weights_only=True)
    return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


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

  ENC_COND_LEN = 6 * S3_SR
  DEC_COND_LEN = 10 * S3GEN_SR

  def __init__(
      self,
      t3: T3,
      s3gen: S3Gen,
      ve: VoiceEncoder,
      conds: Conditionals,
      tokenizer: EnTokenizer,
      seed: int | None = None,
  ):
    self.conds = conds
    self.seed = StableSeed(seed) if seed else contextlib.nullcontext()
    self.tokenizer = tokenizer

    self.s3gen = s3gen
    self.t3 = t3
    self.ve = ve

  @classmethod
  def from_local(
      cls, checkpoint_path: Path, seed: int | None = None, voice: Path | None = None) -> TTS:
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(checkpoint_path / 've.safetensors'))
    ve.to(DEVICE).eval()

    t3 = T3()
    t3_state = load_file(checkpoint_path / 't3_cfg.safetensors')
    if 'model' in t3_state.keys():
      t3_state = t3_state['model'][0]
    t3.load_state_dict(t3_state)
    t3.to(device=DEVICE, dtype=DTYPE).eval()

    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(checkpoint_path / 's3gen.safetensors'), strict=False)
    s3gen.to(device=DEVICE, dtype=torch.float16).eval()  # Doesn't support bfloat16

    s3gen.flow.fp16 = True
    s3gen.mel2wav.to(dtype=torch.float32)
    s3gen.tokenizer.to(dtype=torch.float32)
    s3gen.speaker_encoder.to(dtype=torch.float32)

    tokenizer = EnTokenizer(str(checkpoint_path / 'tokenizer.json'))

    if voice:
      s3gen_ref_wav, _sr = librosa.load(voice, sr=S3GEN_SR)
      ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
      s3gen_ref_wav = s3gen_ref_wav[:cls.DEC_COND_LEN]
      s3gen_ref_dict = s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=DEVICE)

      # Speech cond prompt tokens
      if plen := t3.hp.speech_cond_prompt_len:
        t3_cond_prompt_tokens, _ = s3gen.tokenizer.forward(
            [ref_16k_wav[:cls.ENC_COND_LEN]], max_len=plen)
        t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(DEVICE)

      # Voice-encoder speaker embedding
      ve_embed = torch.from_numpy(ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
      ve_embed = ve_embed.mean(axis=0, keepdim=True).to(DEVICE)

      t3_cond = T3Cond(
          speaker_emb=ve_embed,
          cond_prompt_speech_tokens=t3_cond_prompt_tokens,
          emotion_adv=0.5 * torch.ones(1, 1, 1),
      ).to(device=DEVICE, dtype=DTYPE)
      conds = Conditionals(t3_cond, s3gen_ref_dict)
    elif (builtin_voice := checkpoint_path / 'conds.pt').exists():
      conds = Conditionals.load(builtin_voice).to(DEVICE)

    return cls(t3, s3gen, ve, conds, tokenizer, seed=seed)

  @classmethod
  def from_pretrained(cls, seed: int | None = None, voice: Path | None = None) -> TTS:
    for path in PRETRAINED_FILES:
      local_path = hf_hub_download(repo_id=HF_REPO_ID, filename=path)
    return cls.from_local(Path(local_path).parent, seed=seed, voice=voice)

  def generate(
    self,
    text: str,
    output_path: Path,
    repetition_penalty: float = 1.2,
    min_p: float = 0.05,
    top_p: float = 1.0,
    exaggeration: float = 0.3,
    cfg_weight: float = 0.5,
    temperature: float = 0.5,
  ):
    if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
      _cond: T3Cond = self.conds.t3
      self.conds.t3 = T3Cond(
          speaker_emb=_cond.speaker_emb,
          cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
          emotion_adv=exaggeration * torch.ones(1, 1, 1),
      ).to(device=DEVICE, dtype=DTYPE)

    text_tokens = self.tokenizer.text_to_tokens(clean_text(text)).to(DEVICE)

    if cfg_weight > 0.0:
      text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

    text_tokens = torch.nn.functional.pad(text_tokens, (1, 0), value=self.t3.hp.start_text_token)
    text_tokens = torch.nn.functional.pad(text_tokens, (0, 1), value=self.t3.hp.stop_text_token)

    with torch.inference_mode(), self.seed:
      speech_tokens = self.t3.inference(
          t3_cond=self.conds.t3,
          text_tokens=text_tokens,
          max_new_tokens=1000,  # TODO: use the value in config
          temperature=temperature,
          cfg_weight=cfg_weight,
          repetition_penalty=repetition_penalty,
          min_p=min_p,
          top_p=top_p,
      )

      # Extract only the conditional batch.
      speech_tokens = speech_tokens[0]

      # TODO: output becomes 1D
      speech_tokens = drop_invalid_tokens(speech_tokens)
      speech_tokens = speech_tokens[speech_tokens < 6561]
      speech_tokens = speech_tokens.to(DEVICE)

      wav, _ = self.s3gen.inference(speech_tokens=speech_tokens, ref_dict=self.conds.gen)
      wav = wav.squeeze(0).detach().cpu().numpy()

    torchaudio.save(output_path, torch.from_numpy(wav).unsqueeze(0), S3GEN_SR)


def main(args: argparse.Namespace):
  """Initializes Chatterbox with the given voice, if any, and generates the given text."""
  tts = TTS.from_pretrained(seed=args.seed, voice=args.voice)
  tts.generate(args.text, args.output, exaggeration=args.exaggeration, temperature=args.temperature)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='chatterbox.tts')

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
