import numpy as np


class SplitNormal:
    """
    This class implements a split normal distribution, i.e. a normal distribution with two different standard
    deviations. The chances of falling left or right of the distribution's mean are equal.
    """
    def __init__(self, loc: float, scale_left: float, scale_right: float):
        """
        Initializes a SplitNormal object.

        Parameters
        ----------
        loc : float
            The mean of the distribution.
        scale_left : float
            The left standard deviation of the distribution
        scale_right : float
            The right standard deviation of the distribution
        """
        self.loc = loc
        self.scale_left = scale_left
        self.scale_right = scale_right

    def random(self, size: int) -> np.ndarray:
        """
        Samples the distribution randomly for a certain number of values.

        Parameters
        ----------
        size : int
            Number of values to output.

        Returns
        -------
        samples : np.ndarray
            Randomly sampled values of the given size.
        """
        mask = np.random.randint(0, 2, size) == 1
        mask_sum = mask.sum()
        # The two distributions are sampled using a mean of 0 which allows to fold them to half a distribution and
        # to offset them to self.loc
        left_samples = self.loc - np.abs(np.random.normal(0, self.scale_left, size=(size-mask_sum)))
        right_samples = self.loc + np.abs(np.random.normal(0, self.scale_right, size=mask_sum))
        samples = np.zeros(size)
        samples[~mask] = left_samples
        samples[mask] = right_samples
        return samples
