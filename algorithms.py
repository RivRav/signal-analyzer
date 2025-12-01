import numpy as np
from scipy.signal import find_peaks as scipy_find_peaks, peak_widths


def compute_baseline(signal):
    """
    - estimate a baseline value for a signal using percentile trimming

    @Parameters
    signal : np.ndarray
        - one-dimensional array containing the signal values

    @Returns
    float
        - the median of the trimmed signal, used as a baseline estimate

    @Notes
    - the function trims the lowest 5% and highest 5% of values to reduce the influence of outliers and spikes median calculations
    """
    # get the lower and upper percentile thresholds
    lower, upper = np.percentile(signal, [5, 95])
    # keep only values inside the percentile range
    trimmed = signal[(signal >= lower) & (signal <= upper)]
    # use the median of the trimmed signal as the baseline
    return np.median(trimmed)


def find_peaks(x: np.ndarray,
               baseline: float = None,
               section_sd: float = None,
               section_mean: float = None,
               signal_total_sd: float = None,
               distance: int = 5000,
               prominence: float = 30,
               local_dist: int = 80000) -> tuple[np.ndarray, np.ndarray]:
    """
    - detect and classify peaks in a one-dimensional signal
    - the function first uses scipy.signal.find_peaks to find peaks above a threshold based on baseline and total standard deviation. Each peak is then classified as either a tumor peak or a water peak according to its behaviour

    @Parameters
    x : np.ndarray
        - one-dimensional input signal
    baseline : float, optional
        - baseline level of the signal
    section_sd : float, optional
        - standard deviation of the current section (not used, we kept it for extensibility)
    section_mean : float, optional
        - mean of the current section (not used, kept for similar reasons)
    signal_total_sd : float, optional
        - overall standard deviation of the signal (to build the height threshold)
    distance : int, optional
        - minimum number of samples between consecutive peaks (default 15000)
    prominence : float, optional
        - minimum prominence for a peak to be accepted (default 30)
    local_dist : int, optional
        - half-width of the local window around each peak used for classification (default 20000)

    @Returns
    tuple[np.ndarray, np.ndarray]
        tumor_peaks : np.ndarray
            - indices of peaks classified as tumor peaks
        water_peaks : np.ndarray
            - same but as water peaks

    @Raises
    ValueError
        - not raised explicitly, but any unexpected errors are caught and result in empty arrays being returned

    @Notes
    - classification uses both the local minimum around the peak and the peak height relative to the baseline
    """

    try:
        #use scipy's find_peaks to get init peaks
        peaks, properties = scipy_find_peaks(
            x,
            height=baseline + 0.7 * signal_total_sd,
            distance=distance,
            prominence=prominence
        )

        # extract peak heights/widths from properties if present, otherwise from the signal
        peak_heights = properties.get("peak_heights", x[peaks])
        peak_widths = properties.get("peak_widths", x[peaks])

        tumor_peaks = []
        water_peaks = []
        total_peak_size = 0

        # loop over all peaks and classify them
        for i, peak_index in enumerate(peaks):
            peak_height = peak_heights[i]
            peak_width = peak_widths[i]
            peak_size = peak_height * peak_width * 0.5

            # define a local window around the peak + minimum around it
            start = max(0, peak_index - local_dist)
            end = min(len(x), peak_index + local_dist)
            local_area = x[start:end]

            local_min = min(local_area)

            # if the local minimum is well below the baseline and the peak is not too high -> water peak, otherwise -> tumor peak
            if (local_min < baseline - 2*signal_total_sd) and (peak_height < baseline + 9*signal_total_sd):
                water_peaks.append(peak_index)
            else:
                tumor_peaks.append(peak_index)
                total_peak_size += peak_size

        # return indices
        return np.array(tumor_peaks), np.array(water_peaks)

    except Exception as e:
        # logging in case of any unexpected error and return empty arrays
        print("Error finding peaks:", e)
        return np.array([]), np.array([])


def minmax_downsample(x: np.ndarray,
                      y: np.ndarray,
                      n_bins: int | None = None,
                      canvas_width: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    - downsample a signal for plotting using a min-max binning strategy
    - the function reduces the number of points to be plotted by splitting the signal into bins and keeping only the minimum and maximum values in each -> preserves waveform with reduced rendering costs

    @Parameters
    x : np.ndarray
        - one-dimensional array of x-values (time)
    y : np.ndarray
        - one-dimensional array of y-values (amplitude)
    n_bins : int, optional
        - desired number of bins (none -> from canvas_width)
    canvas_width : int, optional
        - approximate width of the plotting canvas in pixels

    @Returns
    tuple[np.ndarray, np.ndarray]
        x_minmax : np.ndarray
            - downsampled x-values (duplicated for min/max)
        y_minmax : np.ndarray
            - downsampled y-values with minima and maxima

    @Notes
    - if the length of the input data is small (<= 2000 samples), the original arrays are returned unchanged
    """

    n = len(y)

    # if the signal is already small enough -> no downsampling needed
    if n <= 2000:
        return x, y

    # decide how many bins to use
    if n_bins is None:
        n_bins = max(canvas_width or 0, 2000)

    # build bin edges and get min/max y-values from each bin
    bins = np.linspace(0, n, n_bins + 1, dtype=int)
    y_min = np.minimum.reduceat(y, bins[:-1])
    y_max = np.maximum.reduceat(y, bins[:-1])
    # use the midpoint of each bin as x-pos
    x_mid = x[(bins[:-1] + bins[1:]) // 2]

    # output arrays
    out_len = 2 * len(x_mid)
    x_minmax = np.empty(out_len, dtype=x.dtype)
    y_minmax = np.empty(out_len, dtype=y.dtype)

    x_minmax[0::2] = x_mid
    x_minmax[1::2] = x_mid
    y_minmax[0::2] = y_min
    y_minmax[1::2] = y_max

    return x_minmax, y_minmax