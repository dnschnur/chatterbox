"""Utilities shared across models."""

import torchaudio

from torch import Tensor

# Parameters for torchaudio's Resample, favoring quality over speed.
RESAMPLE_PARAMS = {
  'resampling_method': 'sinc_interp_kaiser',
  'lowpass_filter_width': 64,
  'rolloff': 0.9475937167399596,
  'beta': 14.769656459379492,
}


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self


class Audio:
  """Wrapper around an audio tensor that automatically resamples it as needed."""

  def __init__(self, source: Tensor, source_sample_rate: int):
    """Wraps the given reference audio Tensor having the given sample rate."""
    self.source: Tensor = source
    self.source_sample_rate: int = source_sample_rate

    # Cache mapping from sample rate to audio tensor.
    self.references: dict[int, Tensor] = {source_sample_rate: source}

  def __getitem__(self, sample_rate: int) -> Tensor:
    """Returns the reference audio, resampled to the given sample rate."""
    reference = self.references.get(sample_rate)

    if reference is None:
      resampler = torchaudio.transforms.Resample(
          orig_freq=self.source_sample_rate, new_freq=sample_rate, **RESAMPLE_PARAMS
      ).to(self.source.device)
      reference = resampler(self.source)
      self.references[sample_rate] = reference

    return reference
