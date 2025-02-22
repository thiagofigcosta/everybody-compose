from typing import List, Tuple
import numpy as np
import torch

from models.transformer import DeepBeatsTransformer
from models.lstm_full_attn import DeepBeatsAttentionRNN
from models.lstm_local_attn import DeepBeatsLSTMLocalAttn
from models.vanilla_rnn import DeepBeatsVanillaRNN
from utils.distribution import DistributionGenerator, TransformerDistribution, LocalAttnLSTMDistribution, AttentionRNNDistribution, VanillaRNNDistribution
from tqdm import tqdm

def get_distribution_generator(model, beats, device) -> DistributionGenerator:
    """
    - `model`: The model to use for sampling
    - `beats`: a numpy array of shape (seq_len, 2), containing the beats
    - `device`: the device to use
    """
    beats = torch.from_numpy(beats).float().to(device)
    if isinstance(model, DeepBeatsTransformer):
        return TransformerDistribution(model, beats, device)
    elif isinstance(model, DeepBeatsAttentionRNN):
        return AttentionRNNDistribution(model, beats, device)
    elif isinstance(model, DeepBeatsLSTMLocalAttn):
        return LocalAttnLSTMDistribution(model, beats, device)
    elif isinstance(model, DeepBeatsVanillaRNN):
        return VanillaRNNDistribution(model, beats, device)
    else:
        raise NotImplementedError("Sampling is not implemented for this model")

def stochastic_step(prev_note: int, distribution: torch.Tensor, top_p: float = 0.9, top_k: int=4, repeat_decay: float = 0.5, temperature = 1.) -> Tuple[int, float]:
    """
    - `distribution`: a tensor of shape (n_notes, ), containing the conditional distribution of the next note
    - `top_p`: sample only the top p% of the distribution
    - `top_k`: sample only the top k notes of the distribution
    - `repeat_decay`: penalty on repeating the same note. Each time the same note is repeated, the probability of repeating it is multiplied by `1 - repeat_decay`.
                      the probability of getting N repeats is upper bounded by `(1 - repeat_decay) ** N`
    - `temperature`: temperature of the distribution. Lower temperature gives more confidence to the most probable notes, higher temperature gives a more uniform distribution.

    Returns:
    - `sampled_note`: an integer representing the sampled note
    - `conditional_likelihood`: the conditional likelihood of the sampled note: P(note | sampled_sequence). This will be useful for beam search.
    """
    assert distribution.shape[0] == 128
    assert len(distribution.shape) == 1
    assert 0 <= top_p <= 1, "top_p must be between 0 and 1"
    assert 0 <= repeat_decay <= 1, "repeat_decay must be between 0 and 1"
    assert temperature > 0, "temperature must be positive"
    # penalize previous note
    distribution[prev_note] *= (1 - repeat_decay)
    # sample only the top p of the distribution
    sorted_prob, sorted_idx = torch.sort(distribution, descending=True)
    cumsum_prob = torch.cumsum(sorted_prob, dim=0)
    top_p_mask = cumsum_prob < top_p
    top_p_mask[0] = True
    top_p_idx = sorted_idx[top_p_mask][:top_k]
    top_p_distribution = distribution[top_p_idx]
    # normalize the distribution
    top_p_distribution = top_p_distribution / top_p_distribution.sum()
    # apply temperature
    top_p_distribution = top_p_distribution ** (1 / temperature)
    # sample
    sampled_note = int(torch.multinomial(top_p_distribution, 1).item())
    conditional_likelihood = top_p_distribution[sampled_note].item()
    return top_p_idx[sampled_note].item(), conditional_likelihood

def stochastic_search(model, beats: np.ndarray, hint: List[int], device: str, top_p: float= 0.9, top_k:int= 4, repeat_decay: float = 0.5, temperature=1.) -> np.ndarray:
    """
    - `model`: model to use for sampling
    - `seq_len`: the length of the sequence to be sampled
    - `device`: the device to use
    - `top_p`: sample only the top p% of the distribution
    - `top_k`: sample only the top k notes of the distribution
    - `repeat_decay`: penalty on repeating the same note. Each time the same note is repeated, the probability of repeating it is multiplied by `1 - repeat_decay`.
                      the probability of getting N repeats is upper bounded by `(1 - repeat_decay) ** N`
    - `initial_note`: the "sequence-start" placeholder.
    - `temperature`: temperature of the distribution. Lower temperature gives more confidence to the most probable notes, higher temperature gives a more uniform distribution.

    Returns:
    - `generated_sequence`: a numpy array of shape (seq_len, ), containing the generated sequence
    """
    dist = get_distribution_generator(model, beats, device)
    state = dist.initial_state(hint)
    generated_sequence = hint[:]
    prev_note = generated_sequence[-1]
    progress_bar = tqdm(range(beats.shape[0] - len(hint)), desc="Stochastic search")
    for _ in progress_bar:
        # get the distribution
        state, distribution = dist.proceed(state, prev_note)
        # sample
        sampled_note, _ = stochastic_step(prev_note, distribution, top_p, top_k, repeat_decay, temperature)
        generated_sequence.append(sampled_note)
        prev_note = sampled_note
    return np.array(generated_sequence)

def beam_search(model, beats: np.ndarray, hint: List[int], device: str, repeat_decay: float = 0.5, num_beams: int = 3, beam_prob: float = 0.7, temperature=1.) -> np.ndarray:
    """
    - `model`: model to use for sampling
    - `seq_len`: the length of the sequence to be sampled
    - `device`: the device to use
    - `repeat_decay`: penalty on repeating the same note. Each time the same note is repeated, the probability of repeating it is multiplied by `1 - repeat_decay`.
                      the probability of getting N repeats is upper bounded by `(1 - repeat_decay) ** N`
    - `initial_note`: the "sequence-start" placeholder.
    - `num_beams`: number of beams to use

    Returns:
    - `generated_sequence`: a numpy array of shape (seq_len, ), containing the generated sequence
    """
    dist = get_distribution_generator(model, beats, device)
    state = dist.initial_state(hint)
    beams = [(hint[:], state, 0)] # (generated_sequence, state, log_likelihood)
    progress_bar = tqdm(range(beats.shape[0] - len(hint)), desc="Beam search")
    for _ in progress_bar:
        beam_choice = np.random.rand()
        if beam_choice < beam_prob:
            new_beams = []
            for beam in beams:
                prev_note = beam[0][-1]
                state, distribution = dist.proceed(beam[1], prev_note)
                # modify the distribution using the repeat_decay
                distribution[prev_note] *= (1 - repeat_decay)
                # sample
                for sampled_note in range(128):
                    new_beam = (beam[0] + [sampled_note], state, beam[2] + np.log(distribution[sampled_note].item()))
                    new_beams.append(new_beam)
            # sort the beams by their likelihood
            new_beams = sorted(new_beams, key=lambda x: x[2], reverse=True)
            # keep only the top num_beams
            beams = new_beams[:num_beams]
        else:
            new_beams = []
            for beam in beams:
                prev_note = beam[0][-1]
                state, distribution = dist.proceed(beam[1], prev_note)
                # sample
                sampled_note, conditional_likelihood = stochastic_step(prev_note, distribution, 1.0, 128, repeat_decay, temperature)
                new_beam = (beam[0] + [sampled_note], state, beam[2] + np.log(conditional_likelihood))
                new_beams.append(new_beam)
            # sort the beams by their likelihood
            new_beams = sorted(new_beams, key=lambda x: x[2], reverse=True)
            # keep only the top num_beams
            beams = new_beams[:num_beams]
    # return the beam with the highest likelihood
    return np.array(beams[0][0])
    


