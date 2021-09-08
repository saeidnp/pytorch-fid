import torch
import numpy as np
from abc import ABC, abstractmethod

class GeneratorBase(ABC):
    def __init__(self, N, max_batch_size):
        """This class wraps around a generative model and most importantly
           defines a sampling and a loader function to later be used for 
           computing FID scores.
           The sampling function provides a convenient way to draw an
           arbitrary-sized batch of samples.
           The loader function provides a dataloader-like object using
           the generative model.

        Args:
            max_batch_size (int): Maximum batch size to ask from the generative model in one shot
            N (int): The (synthetic) dataset size. It will be used in the loader function
        """
        self.max_batch_size = max_batch_size
        self.N = N

    @torch.no_grad()
    def sample_batch(self, batch_size):
        # "microbatch_size_list" is a sequence of microbatch sizes such that, once all concatenated,
        # creates a batch of size "batch_size"
        microbatch_size_list = np.arange(0, batch_size, self.max_batch_size, dtype=np.int)
        microbatch_size_list = np.concatenate([microbatch_size_list, [batch_size]])
        microbatch_size_list = microbatch_size_list[1:] - microbatch_size_list[:-1]
        assert sum(microbatch_size_list) == batch_size, "Internal error in creating the sequence of microbatch sizes"
        batch = torch.cat([self.sample(b) for b in microbatch_size_list], dim=0)
        return batch

    @abstractmethod
    def sample(self, batch_size):
        """Should return a batch of samples from the model.

        Args:
            batch_size (int)

        Returns:
            batch: A batch of size batch_size * sample_shape.
                Each samples should be an image with pixel values in [0, 1]
        """
        return None

    def loader(self, batch_size):
        """Yields batches of size "batch_size" until producing a total of n samples.
        """
        cnt = 0
        while cnt < self.N:
            batch = self.sample_batch(min(batch_size, self.N - cnt))
            cnt += len(batch)
            yield batch
        raise StopIteration