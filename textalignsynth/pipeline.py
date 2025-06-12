from typing import List, Tuple
from tqdm import tqdm

import numpy as np
import librosa
import libtsm
import pyloudnorm as pyln
from .loudness import LoudnessMeter
from .synthesis import TTSWrapper


class Commenter:
    def __init__(self, device: str = "cpu", language="en", tts_verbose=False):
        """
        This module can be used for adding synthesized text comments to an audio file.

        Parameters:
            device          device to be used for text-to-speech synthesis (e.g. "cpu" or "cuda")
            language        language of the text comments;
                                currently supported: english ("en"), german ("de")
            tts_verbose     verbose mode
        """
        self.synthesizer = TTSWrapper(device=device, language=language, verbose=tts_verbose)
        self.loudness_meter = LoudnessMeter(self.synthesizer.fs)

    def __call__(
        self,
        x_orig: np.ndarray,
        x_fs: float,
        comments: List[Tuple[float, str]],
        speed: float = 1.0,
        t_min: float = None,
        t_max: float = None,
        pos_rel: float = 0.0,
        pos_offset_abs: float = 0.0,
        offset_loc: float = -5.0,
        offset_glob: float = -5.0,
        w_glob_loc: float = 0.5,
        return_comment_track: bool = False,
        show_progress_bar: bool = True,
    ):
        """
        This is the main function which synthesizes every comment individually and then mixes the original audio
        signal with the comments

        Parameters:
            x_orig                      original audio signal
            comments                    comment list of the form [[float: time in seconds, str: comment], ...],
                                            e.g., [[0.4, 'G. minor.'], [1.3 'B. flat minor.']]
            x_fs                          sampling frequency of x_orig
            speed                       determines the speed of the synthesized comments (float, no alteration for speed=1.0)
            duration_range              minimum and maximum duration per comment in seconds;
                                            no changes for too short/long comments if the respective entry is None
            pos_rel                     relative positioning of the comments w.r.t. the given time instants t;
                                            i.e. 0 = start comment at time t, 0.5 = center-align comment with time t, 1 = end comment at time t
            pos_offset_abs              absolute temporal offset in seconds which is applied to every comment in the list
            loudness_global_offset_db   offset in dB compared to the original audio's global loudness
            loudness_local_offset_db    offset in dB compared to the original audio's short-term loudness
            alpha_global_local          factor in the range [0, 1] for the tradeoff between global and local loudness
                                            i.e. 0 = only consider global loudness, 1 = only consider local loudness
            return_comment_track        whether to return the comment track in addition to the commented audio signal
        """
        # resample audio signal to sampling rate of TTS model
        x_resampled = librosa.resample(x_orig, orig_sr=x_fs, target_sr=self.synthesizer.fs)

        # determine global loudness
        x_loudness_global = self.loudness_meter.measure_loudness(x_resampled)

        # initialize comment track
        comment_track = np.zeros_like(x_resampled)
        padding_left = 0

        iterable = tqdm(comments) if show_progress_bar else comments

        # synthesize and post-process comments individually
        for t_m, comment in iterable:
            # synthesize text
            c_m = self.synthesizer(comment)

            # remove leading and trailing silence in comment
            c_m = librosa.effects.trim(np.array(c_m))[0]

            c_m = self._modify_comment_duration(c_m, speed, t_min, t_max)
            n_m = self._get_comment_start(c_m, t_m, pos_rel, pos_offset_abs) + padding_left

            n_m_end = n_m + len(c_m)

            # handle negative n_m
            if n_m < 0:
                padding_left_add = np.abs(n_m) - padding_left
                comment_track = np.pad(
                    comment_track,
                    pad_width=(padding_left_add, 0),
                    mode="constant",
                    constant_values=0.0,
                )
                x_resampled = np.pad(
                    x_resampled,
                    pad_width=(padding_left_add, 0),
                    mode="constant",
                    constant_values=0.0,
                )
                n_m += padding_left_add
                n_m_end += padding_left_add
                padding_left += padding_left_add

            # handle comments exceeding signal duration
            if n_m_end >= len(comment_track):
                padding_right_add = n_m_end - (len(comment_track) - 1)
                comment_track = np.pad(
                    comment_track,
                    pad_width=(0, padding_right_add),
                    mode="constant",
                    constant_values=0.0,
                )
                x_resampled = np.pad(
                    x_resampled,
                    pad_width=(0, padding_right_add),
                    mode="constant",
                    constant_values=0.0,
                )

            # modify loudness
            x_loudness_local = self.loudness_meter.measure_loudness(x_resampled[n_m:n_m_end])
            c_m_loudness = self.loudness_meter.measure_loudness(c_m)
            c_m_loudness_target = w_glob_loc * (x_loudness_local + offset_loc) + (
                1 - w_glob_loc
            ) * (x_loudness_global + offset_glob)
            c_m_normalized = pyln.normalize.loudness(c_m, c_m_loudness, c_m_loudness_target)

            comment_track[n_m:n_m_end] += c_m_normalized

        # superposition
        x_commented = x_resampled + comment_track

        # resampling to original sampling rate
        x_commented = librosa.resample(x_commented, orig_sr=self.synthesizer.fs, target_sr=x_fs)

        if return_comment_track:
            comment_track = librosa.resample(
                comment_track, orig_sr=self.synthesizer.fs, target_sr=x_fs
            )
            return x_commented, comment_track
        else:
            return x_commented

    def _modify_comment_duration(self, comment_wav, speed, t_min, t_max):
        assert (
            t_min is None or t_max is None or t_min <= t_max
        ), "t_min needs to be less or equal to t_max"

        alpha = 1 / speed
        t_comment = alpha * len(comment_wav) / self.synthesizer.fs

        if (t_min is not None) and (t_comment < t_min):
            alpha *= t_min / t_comment
        elif (t_max is not None) and (t_comment > t_max):
            alpha *= t_max / t_comment

        if alpha != 1.0:
            comment_wav = np.squeeze(
                libtsm.tsm.hps_tsm(comment_wav, alpha, Fs=self.synthesizer.fs), axis=1
            )

        return comment_wav

    def _get_comment_start(self, comment_wav, t_start, pos_rel, pos_offset_abs):
        n_start = (t_start + pos_offset_abs) * self.synthesizer.fs - pos_rel * len(comment_wav)
        return round(n_start)
