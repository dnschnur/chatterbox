# Adapted from https://github.com/CorentinJ/Real-Time-Voice-Cloning
# MIT License

import numpy as np
import torch

from torch import Tensor

from ..utils import Audio

from .config import VoiceEncConfig
from .melspec import melspectrogram


def pack(tensors: list[Tensor], seq_len: int=None, pad_value=0):
    """
    Given a list of tensors of length B, of shapes (Ti, ...), packs them in a single tensor of
    shape (B, T, ...) by padding each individual tensor on the right.

    :param tensors: a list of tensors of matching shapes except for the first axis.
    :param seq_len: the value of T. It must be the maximum of the lengths Ti of the tensors at
        minimum. Will default to that value if None.
    :param pad_value: the value to pad the tensors with.
    :return: a (B, T, ...) tensor
    """
    if seq_len is None:
        seq_len = max(len(tensor) for tensor in tensors)
    else:
        assert seq_len >= max(len(tensor) for tensor in tensors)

    device = tensors[0].device
    dtype = tensors[0].dtype

    # Fill the packed tensor with the tensor data
    packed_shape = (len(tensors), seq_len, *tensors[0].shape[1:])
    packed_tensor = torch.full(packed_shape, pad_value, dtype=dtype, device=device)

    for i, tensor in enumerate(tensors):
        packed_tensor[i, :tensor.size(0)] = tensor

    return packed_tensor


def get_num_wins(
    n_frames: int,
    step: int,
    min_coverage: float,
    hp: VoiceEncConfig,
):
    assert n_frames > 0
    win_size = hp.ve_partial_frames
    n_wins, remainder = divmod(max(n_frames - win_size + step, 0), step)
    if n_wins == 0 or (remainder + (win_size - step)) / win_size >= min_coverage:
        n_wins += 1
    target_n = win_size + step * (n_wins - 1)
    return n_wins, target_n


def get_frame_step(
    overlap: float,
    rate: float,
    hp: VoiceEncConfig,
):
    # Compute how many frames separate two partial utterances
    assert 0 <= overlap < 1
    if rate is None:
        frame_step = int(np.round(hp.ve_partial_frames * (1 - overlap)))
    else:
        frame_step = int(np.round((hp.sample_rate / rate) / hp.ve_partial_frames))
    assert 0 < frame_step <= hp.ve_partial_frames
    return frame_step


class VoiceEncoder(torch.nn.Module):
    def __init__(self, hp=VoiceEncConfig()):
        super().__init__()

        self.hp = hp

        # Network definition
        self.lstm = torch.nn.LSTM(self.hp.num_mels, self.hp.ve_hidden_size, num_layers=3, batch_first=True)
        if hp.flatten_lstm_params:
            self.lstm.flatten_parameters()
        self.proj = torch.nn.Linear(self.hp.ve_hidden_size, self.hp.speaker_embed_size)

        # Cosine similarity scaling (fixed initial parameter values)
        self.similarity_weight = torch.nn.Parameter(Tensor([10.]), requires_grad=True)
        self.similarity_bias = torch.nn.Parameter(Tensor([-5.]), requires_grad=True)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, mels: torch.FloatTensor):
        """
        Computes the embeddings of a batch of partial utterances.

        :param mels: a batch of unscaled mel spectrograms of same duration as a float32 tensor
        of shape (B, T, M) where T is hp.ve_partial_frames
        :return: the embeddings as a float32 tensor of shape (B, E) where E is
        hp.speaker_embed_size. Embeddings are L2-normed and thus lay in the range [-1, 1].
        """
        if self.hp.normalized_mels and (mels.min() < 0 or mels.max() > 1):
            raise Exception(f'Mels outside [0, 1]. Min={mels.min()}, Max={mels.max()}')

        # Pass the input through the LSTM layers
        _, (hidden, _) = self.lstm(mels)

        # Project the final hidden state
        raw_embeds = self.proj(hidden[-1])
        if self.hp.ve_final_relu:
            raw_embeds = torch.nn.functional.relu(raw_embeds)

        # L2 normalize the embeddings.
        return raw_embeds / torch.linalg.norm(raw_embeds, dim=1, keepdim=True)

    def inference(self, mels: Tensor, mel_lens, overlap=0.5, rate: float=None, min_coverage=0.8, batch_size=None):
        """
        Computes the embeddings of a batch of full utterances with gradients.

        :param mels: (B, T, M) unscaled mels
        :return: (B, E) embeddings on CPU
        """
        mel_lens = mel_lens.tolist() if torch.is_tensor(mel_lens) else mel_lens

        # Compute where to split the utterances into partials
        frame_step = get_frame_step(overlap, rate, self.hp)
        n_partials, target_lens = zip(*(get_num_wins(l, frame_step, min_coverage, self.hp) for l in mel_lens))

        # Possibly pad the mels to reach the target lengths
        len_diff = max(target_lens) - mels.size(1)
        if len_diff > 0:
            pad = torch.full((mels.size(0), len_diff, self.hp.num_mels), 0, dtype=torch.float32)
            mels = torch.cat((mels, pad.to(mels.device)), dim=1)

        # Group all partials together so that we can batch them easily
        partials = [
            mel[i * frame_step: i * frame_step + self.hp.ve_partial_frames]
            for mel, n_partial in zip(mels, n_partials) for i in range(n_partial)
        ]
        assert all(partials[0].shape == partial.shape for partial in partials)
        partials = torch.stack(partials)

        # Forward the partials
        n_chunks = int(np.ceil(len(partials) / (batch_size or len(partials))))
        partial_embeds = torch.cat([self(batch) for batch in partials.chunk(n_chunks)], dim=0).cpu()

        # Reduce the partial embeds into full embeds and L2-normalize them
        slices = np.concatenate(([0], np.cumsum(n_partials)))
        raw_embeds = [torch.mean(partial_embeds[start:end], dim=0) for start, end in zip(slices[:-1], slices[1:])]
        raw_embeds = torch.stack(raw_embeds)
        embeds = raw_embeds / torch.linalg.norm(raw_embeds, dim=1, keepdim=True)

        return embeds

    @staticmethod
    def utt_to_spk_embed(utt_embeds: Tensor) -> Tensor:
        """
        Takes a tensor of L2-normalized utterance embeddings, computes the mean embedding, and
        L2-normalize it to get a speaker embedding.
        """
        assert utt_embeds.dim() == 2
        utt_embeds = torch.mean(utt_embeds, axis=0)
        return utt_embeds / torch.linalg.norm(utt_embeds, 2)

    def embeds_from_mels(
        self, mels: list[Tensor], mel_lens=None, as_spk=False, batch_size=32, **kwargs
    ) -> Tensor:
        """
        Convenience function for deriving utterance or speaker embeddings from mel spectrograms.

        :param mels: unscaled mels strictly within [0, 1] as either a (B, T, M) tensor or a list of (Ti, M) arrays.
        :param mel_lens: if passing mels as a tensor, individual mel lengths
        :param as_spk: whether to return utterance embeddings or a single speaker embedding
        :param kwargs: args for inference()

        :returns: embeds as a (B, E) float32 numpy array if <as_spk> is False, else as a (E,) array
        """
        # Load mels in memory and pack them
        assert all(mel.shape[1] == mels[0].shape[1] for mel in mels), 'Mels aren\'t in (B, T, M) format'
        mel_lens = [mel.shape[0] for mel in mels]
        mels = pack(mels).to(self.device)

        with torch.inference_mode():
            utt_embeds = self.inference(mels, mel_lens, batch_size=batch_size, **kwargs)

        return self.utt_to_spk_embed(utt_embeds) if as_spk else utt_embeds

    def embeds_from_wav(
        self, reference: Audio, as_spk: bool = False, batch_size: int = 32
    ) -> Tensor:
        """Returns utterance or speaker embeddings from a list of audio tensors."""
        audio = reference[self.hp.sample_rate].to(self.device)
        mels = [melspectrogram(audio, self.hp).T]

        # The rate=1.3 is Resemble's default value.
        return self.embeds_from_mels(mels, as_spk=as_spk, batch_size=batch_size, rate=1.3)
