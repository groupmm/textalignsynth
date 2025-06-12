from pyloudnorm.iirfilter import IIRfilter
import numpy as np
import warnings

"Most code in here is a slightly modified version of the pyloudnorm package https://github.com/csteinmetz1/pyloudnorm"


class LoudnessMeter(object):
    """A loudness meter to measure the loudness of signals according to BS.1770
    Provides a single function "measure_loudness" for the user, which allows for arbitrarily short signals and returns a loudness value in LUFS
    Note that if the signal is shorter than the standard time window of 400ms no gating can be applied and the momentary loudness is returned for that signal
    """

    def __init__(self, rate):
        self.rate = rate
        self.block_size = 0.4  # standard 400ms blocksize of BS.1770
        self.filter_class = "K-weighting"

    def measure_loudness(self, data, threshold=-70.0):
        """Measure the loudness of a signal.

        Uses the weighting filters and block size defined by the meter
        the integrated loudness is measured based upon the gating algorithm
        defined in the ITU-R BS.1770-4 specification.
        If the input signal is short than the standard 400ms block size, the ungated momentary loudness is returned

        Input data must have shape (samples, ch) or (samples,) for mono audio.
        Supports up to 5 channels and follows the channel ordering:
        [Left, Right, Center, Left surround, Right surround]

        Params
        -------
            data : ndarray
            Input multichannel audio data.
            threshold : float
            absolute loudness threshold; this value will be returned if the actual loudness is below the threshold

        Returns
        -------
            LUFS : float
            Integrated gated loudness of the input measured in LUFS or the ungated momentary loudness for signals shorter than the block size of 400ms
        """
        input_data = data
        valid_audio(input_data, self.rate)

        if input_data.ndim == 1:
            input_data = np.reshape(input_data, (input_data.shape[0], 1))

        numSamples = input_data.shape[0]

        if numSamples > (self.block_size * self.rate):
            return max(self._integrated_loudness(data), threshold)
        else:
            mom_loudness, _ = self._momentary_loudness(data)
            return max(
                mom_loudness, threshold
            )  # for a signal duration < blocksize we only get a single momentary loudness value

    def _integrated_loudness(self, data):
        """Measure the integrated gated loudness of a signal.

        Uses the weighting filters and block size defined by the meter
        the integrated loudness is measured based upon the gating algorithm
        defined in the ITU-R BS.1770-4 specification.

        Input data must have shape (samples, ch) or (samples,) for mono audio.
        Supports up to 5 channels and follows the channel ordering:
        [Left, Right, Center, Left surround, Right surround]

        Params
        -------
        data : ndarray
            Input multichannel audio data.

        Returns
        -------
        LUFS : float
            Integrated gated loudness of the input measured in dB LUFS.
        """
        input_data = data.copy()
        valid_audio(input_data, self.rate)

        if input_data.ndim == 1:
            input_data = np.reshape(input_data, (input_data.shape[0], 1))

        numChannels = input_data.shape[1]

        mom_loudness, z = self._momentary_loudness(input_data)

        Gamma_a = -70.0  # -70 LKFS = absolute loudness threshold
        G = [1.0, 1.0, 1.0, 1.41, 1.41]  # channel gains

        # find gating block indices above absolute threshold
        J_g = [j for j, l_j in enumerate(mom_loudness) if l_j >= Gamma_a]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # calculate the average of z[i,j] as show in eq. 5
            z_avg_gated = [np.mean([z[i, j] for j in J_g]) for i in range(numChannels)]
        # calculate the relative threshold value (see eq. 6)
        Gamma_r = (
            -0.691
            + 10.0 * np.log10(np.sum([G[i] * z_avg_gated[i] for i in range(numChannels)]))
            - 10.0
        )

        # find gating block indices above relative and absolute thresholds  (end of eq. 7)
        J_g = [j for j, l_j in enumerate(mom_loudness) if (l_j > Gamma_r and l_j > Gamma_a)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # calculate the average of z[i,j] as show in eq. 7 with blocks above both thresholds
            z_avg_gated = np.nan_to_num(
                np.array([np.mean([z[i, j] for j in J_g]) for i in range(numChannels)])
            )

        # calculate final loudness gated loudness (see eq. 7)
        with np.errstate(divide="ignore"):
            LUFS = -0.691 + 10.0 * np.log10(
                np.sum([G[i] * z_avg_gated[i] for i in range(numChannels)])
            )

        return LUFS

    def _momentary_loudness(self, data):
        input_data = data.copy()
        valid_audio(input_data, self.rate)

        if input_data.ndim == 1:
            input_data = np.reshape(input_data, (input_data.shape[0], 1))

        numChannels = input_data.shape[1]
        numSamples = input_data.shape[0]
        # Apply frequency weighting filters - account for the acoustic response of the head and auditory system
        for filter_class, filter_stage in self._filters.items():
            for ch in range(numChannels):
                input_data[:, ch] = filter_stage.apply_filter(input_data[:, ch])

        G = [1.0, 1.0, 1.0, 1.41, 1.41]  # channel gains
        T_g = self.block_size  # 400 ms gating block standard
        if numSamples > (self.block_size * self.rate):
            overlap = 0.75  # overlap of 75% of the block duration
            step = 1.0 - overlap  # step size by percentage

            T = numSamples / self.rate  # length of the input in seconds
            numBlocks = int(
                np.round(((T - T_g) / (T_g * step))) + 1
            )  # total number of gated blocks (see end of eq. 3)
            j_range = np.arange(0, numBlocks)  # indexed list of total blocks
            z = np.zeros(shape=(numChannels, numBlocks))  # instantiate array - trasponse of input

            for i in range(numChannels):  # iterate over input channels
                for j in j_range:  # iterate over total frames
                    lower = int(
                        T_g * (j * step) * self.rate
                    )  # lower bound of integration (in samples)
                    upper = int(
                        T_g * (j * step + 1) * self.rate
                    )  # upper bound of integration (in samples)
                    # caluate mean square of the filtered for each block (see eq. 1)
                    z[i, j] = (1.0 / (T_g * self.rate)) * np.sum(
                        np.square(input_data[lower:upper, i])
                    )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # loudness for each jth block (see eq. 4)
                lower = [
                    -0.691 + 10.0 * np.log10(np.sum([G[i] * z[i, j] for i in range(numChannels)]))
                    for j in j_range
                ]
        else:
            # only a single block
            z = np.zeros(shape=(numChannels))
            for i in range(numChannels):  # iterate over input channels
                z[i] = (1.0 / (T_g * self.rate)) * np.sum(np.square(input_data[:, i]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # single momentary loudness value
                lower = -0.691 + 10.0 * np.log10(np.sum(G * z))

        return lower, z

    @property
    def filter_class(self):
        return self._filter_class

    @filter_class.setter
    def filter_class(self, value):
        self._filters = {}  # reset (clear) filters
        # K-weighting filter
        self._filters["high_shelf"] = IIRfilter(
            4.0, 1 / np.sqrt(2), 1500.0, self.rate, "high_shelf"
        )
        self._filters["high_pass"] = IIRfilter(0.0, 0.5, 38.0, self.rate, "high_pass")


def valid_audio(data, rate):
    """Validate input audio data.

    Ensure input is numpy array of floating point data bewteen -1 and 1

    Taken from pyloudnorm package but dropping the blocksize check

    Params
    -------
    data : ndarray
        Input audio data
    rate : int
        Sampling rate of the input audio in Hz

    Returns
    -------
    valid : bool
        True if valid audio

    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be of type numpy.ndarray.")

    if not np.issubdtype(data.dtype, np.floating):
        raise ValueError("Data must be floating point.")

    if data.ndim == 2 and data.shape[1] > 5:
        raise ValueError("Audio must have five channels or less.")

    return True
