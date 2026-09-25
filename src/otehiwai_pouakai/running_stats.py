import numpy as np

class RunningStats:
    """
    Per-pixel running mean/standard-deviation accumulator using Welford's
    online algorithm.

    Useful for combining a stack of same-shaped images (e.g. building a
    master dark/flat via sigma-clipped combination) without holding every
    frame in memory at once: call `add` once per frame, then `mean()`/
    `std()` at the end for the per-pixel statistics across everything
    added so far.

    Welford's algorithm is used instead of the simpler "sum and
    sum-of-squares" formula (variance = E[x^2] - E[x]^2) because that
    formula is numerically unstable whenever the mean is large relative
    to the standard deviation -- e.g. a CCD bias level of several
    thousand ADU against a read-noise std of only a few ADU -- since it
    subtracts two large, nearly-equal numbers. Welford's algorithm
    instead tracks the running mean and the running sum of squared
    deviations from that mean (`M2`) directly, avoiding that cancellation
    entirely, regardless of the absolute signal level.

    Parameters
    ----------
    shape : tuple of int
        Shape of the per-pixel arrays being accumulated over.
    dtype : numpy dtype
        Output dtype for `mean()`/`std()`. Internal accumulators are
        always kept in float64 regardless of this setting, since that
        precision is exactly what Welford's algorithm is meant to
        preserve; `dtype` only affects the returned arrays.
    """

    def __init__(self, shape, dtype=np.float64):
        self._shape = shape
        self._out_dtype = dtype

        self.count = np.zeros(shape, dtype=np.uint32)
        self._mean = np.zeros(shape, dtype=np.float64)
        self._m2 = np.zeros(shape, dtype=np.float64)

    def add(self, data, mask=None):
        """
        Add one frame's worth of per-pixel values to the running
        statistics.

        Parameters
        ----------
        data : ndarray, same shape as `shape`
            Values to add.
        mask : bool ndarray or None
            Which pixels to include from this frame (e.g. to exclude
            pixels rejected by sigma-clipping for this particular frame).
            Defaults to `np.isfinite(data)` if not given, so NaNs are
            automatically excluded.
        """
        if mask is None:
            mask = np.isfinite(data)

        valid = mask
        x = np.asarray(data, dtype=np.float64)

        self.count[valid] += 1
        n = self.count[valid].astype(np.float64)

        delta = x[valid] - self._mean[valid]
        self._mean[valid] += delta / n

        delta2 = x[valid] - self._mean[valid]
        self._m2[valid] += delta * delta2

    def mean(self):
        """
        Return the per-pixel running mean, as an array of shape `shape`
        and dtype `dtype`. Pixels with zero contributing frames are NaN.
        """
        out = np.full(self._shape, np.nan, dtype=self._out_dtype)
        valid = self.count > 0
        out[valid] = self._mean[valid].astype(self._out_dtype)
        return out

    def std(self):
        """
        Return the per-pixel running standard deviation, as an array of
        shape `shape` and dtype `dtype`. Pixels with fewer than two
        contributing frames are NaN (standard deviation is undefined for
        a single sample).
        """
        out = np.zeros(self._shape, dtype=self._out_dtype)
        valid = self.count > 1

        n = self.count[valid].astype(np.float64)
        variance = self._m2[valid] / n
        variance = np.maximum(variance, 0.0)  # guard tiny negative fp noise

        out[valid] = np.sqrt(variance).astype(self._out_dtype)
        out[~valid] = np.nan

        return out

    # Convenience read-only properties reconstructing the equivalent
    # "sum" and "sum of squares" quantities from the Welford
    # accumulators, for any external code that expects them directly.
    @property
    def sum(self):
        return self._mean * self.count.astype(np.float64)

    @property
    def sum_sq(self):
        return self._m2 + self.count.astype(np.float64) * self._mean**2