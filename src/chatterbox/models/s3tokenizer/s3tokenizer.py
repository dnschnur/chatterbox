import torch
import torchaudio

from torch import Tensor

from s3tokenizer.utils import padding
from s3tokenizer.model_v2 import S3TokenizerV2, ModelConfig

# Sampling rate of the inputs to S3TokenizerV2
S3_SR = 16_000
S3_HOP = 160  # 100 frames/sec
S3_TOKEN_HOP = 640  # 25 tokens/sec
S3_TOKEN_RATE = 25
SPEECH_VOCAB_SIZE = 6561


class S3Tokenizer(S3TokenizerV2):

  ignore_state_dict_missing = ("_mel_filters", "window")

  def __init__(self, name: str="speech_tokenizer_v2_25hz", config: ModelConfig = ModelConfig()):
    super().__init__(name)

    self.n_fft = 400
    _mel_filters = torchaudio.functional.melscale_fbanks(
        n_freqs=int(self.n_fft // 2 + 1),
        f_min=0.0,
        f_max=S3_SR / 2.0,
        n_mels=config.n_mels,
        sample_rate=S3_SR,
        norm="slaney",
        mel_scale="slaney").T
    self.register_buffer("_mel_filters", _mel_filters)
    self.register_buffer("window", torch.hann_window(self.n_fft))

  @torch.no_grad()
  def forward(self, audio: Tensor, max_len: int | None = None) -> tuple[Tensor, torch.LongTensor]:
    """
    NOTE: mel-spec has a hop size of 160 points (100 frame/sec).

    Args:
      audio: 16 kHz speech audio tensor.
      max_len: Max length to truncate the output sequence to (25 token/sec).
    """
    if audio.dim() == 1:
      audio = audio.unsqueeze(0)

    mel = self.log_mel_spectrogram(audio)  # [B=1, F, T]
    if max_len is not None:
      mel = mel[..., :max_len * 4]  # num_mel_frames = 4 * num_tokens

    mels, mel_lens = padding([mel.squeeze(0)])
    speech_tokens, speech_token_lens = self.quantize(mels, mel_lens.to(self.device))
    return speech_tokens.long().detach(), speech_token_lens.long().detach()

  def log_mel_spectrogram(self, audio: Tensor, padding: int = 0) -> Tensor:
    """Returns the log-Mel spectrogram of the given audio tensor.

    Args:
      audio: Tensor containing the audio waveform in 16 kHz
      padding: Number of zero samples to pad to the right

    Returns: Tensor with shape = (128, n_frames) that contains the Mel spectrogram.
    """
    audio = audio.to(self.device)
    if padding > 0:
      audio = torch.nn.functional.pad(audio, (0, padding))
    stft = torch.stft(
        audio, self.n_fft, S3_HOP, window=self.window.to(self.device), return_complex=True)
    magnitudes = stft[..., :-1].abs()**2

    mel_spec = self._mel_filters.to(self.device) @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0
