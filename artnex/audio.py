# MIT License

# Copyright (c) 2023 Yuan-Man

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import math
import numpy as np
import collections
import librosa


class AudioSegmenterLocator:
    def __init__(self, energy_threshold=0.1, segment_duration=1.0):
        """
        Initializes the class for audio segmentation and localization.

        Parameters:
        - energy_threshold (float): Energy threshold for segmenting audio.
        - segment_duration (float): Target duration for each audio segment (in seconds).
        """
        self.energy_threshold = energy_threshold
        self.segment_duration = segment_duration

    def segment_and_locate(self, audio_path):
        """
        Segments and locates audio based on energy threshold.

        Parameters:
        - audio_path (str): Path to the audio file.

        Returns:
        - list: List of dictionaries containing segment start and end times.
        """
        try:
            # Load the audio file
            y, sr = librosa.load(audio_path, sr=None)

            # Calculate energy of the audio signal
            energy = np.sum(np.abs(y)**2)

            # Perform audio segmentation based on energy threshold
            segments = []
            start_time = 0.0
            current_segment = {'start': start_time, 'end': start_time}

            for i, amp in enumerate(y):
                current_time = i / sr

                if current_time - start_time >= self.segment_duration or energy <= self.energy_threshold:
                    # End the current segment and start a new one
                    current_segment['end'] = current_time
                    segments.append(current_segment)

                    # Reset for the next segment
                    start_time = current_time
                    current_segment = {'start': start_time, 'end': start_time}
                    energy = 0.0
                else:
                    # Update energy for the current segment
                    energy += np.abs(amp)**2

            # Add the last segment
            current_segment['end'] = current_time
            segments.append(current_segment)

            return segments

        except Exception as e:
            print(f"Error segmenting and locating audio: {e}")
            return None

NEGATIVE_INFINITY = -float("inf")

class CTCDecoder:
    def __init__(self, beam_size=10, blank=0):
        """
        Initializes the CTCDecoder.

        Parameters:
        - beam_size (int): The size of the beam during decoding.
        - blank (int): Index of the CTC blank label.
        """
        self.beam_size = beam_size
        self.blank = blank

    def make_new_beam(self):
        """
        Creates a new beam for storing probabilities of candidate sequences.

        Returns:
        - defaultdict: The new beam.
        """
        return collections.defaultdict(lambda: (NEGATIVE_INFINITY, NEGATIVE_INFINITY))

    def logsumexp(self, *args):
        """
        Stable log-sum-exp operation.

        Parameters:
        - *args: Input values.

        Returns:
        - float: Result of log-sum-exp operation.
        """
        if all(a == NEGATIVE_INFINITY for a in args):
            return NEGATIVE_INFINITY
        a_max = max(args)
        lsp = math.log(sum(math.exp(a - a_max) for a in args))
        return a_max + lsp

    def decode(self, probs):
        """
        Performs decoding for the given output probabilities.

        Parameters:
        - probs (numpy.ndarray): Output probabilities (post-softmax). Shape: (time x output dim).

        Returns:
        - tuple: Decoded sequence and negative log-likelihood.
        """
        T, S = probs.shape
        probs = np.log(probs)

        # Initialize the beam with an empty sequence and probabilities.
        beam = [(tuple(), (0.0, NEGATIVE_INFINITY))]

        for t in range(T):
            next_beam = self.make_new_beam()

            for s in range(S):
                p = probs[t, s]

                for prefix, (p_b, p_nb) in beam:
                    # If choosing a blank label
                    if s == self.blank:
                        n_p_b, n_p_nb = next_beam[prefix]
                        n_p_b = self.logsumexp(n_p_b, p_b + p, p_nb + p)
                        next_beam[prefix] = (n_p_b, n_p_nb)
                        continue

                    end_t = prefix[-1] if prefix else None
                    n_prefix = prefix + (s,)
                    n_p_b, n_p_nb = next_beam[n_prefix]

                    if s != end_t:
                        n_p_nb = self.logsumexp(n_p_nb, p_b + p, p_nb + p)
                    else:
                        n_p_nb = self.logsumexp(n_p_nb, p_b + p)

                    next_beam[n_prefix] = (n_p_b, n_p_nb)

                    if s == end_t:
                        n_p_b, n_p_nb = next_beam[prefix]
                        n_p_nb = self.logsumexp(n_p_nb, p_nb + p)
                        next_beam[prefix] = (n_p_b, n_p_nb)

            # Sort and trim the beam before moving to the next time-step.
            beam = sorted(next_beam.items(), key=lambda x: self.logsumexp(*x[1]), reverse=True)[:self.beam_size]

        # Return the best sequence and negative log-likelihood.
        best = beam[0]
        return best[0], -self.logsumexp(*best[1])


