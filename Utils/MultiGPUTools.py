import numpy as np
import pandas as pd
import torch


def pad_size(n_examples, n_gpus, batch_size):
    # Returns the number of tiles that must be appended to a dataset of length n_examples
    # to be compatible with all_gather for multi-GPU inference.    

    # When processing on multiple GPUs, the effective batch_size becomes n_gpus * batch_size:
    effective_batch_size = n_gpus * batch_size

    # Calculate the number of effective batches that can be done with the current dataset:
    neb = np.floor(n_examples / effective_batch_size).astype(int)

    # Get the corresponding number of examples:
    n_full_examples = neb * effective_batch_size

    # Calculate the number of remaining examples that would be left out:
    n_left_out = n_examples - n_full_examples.astype(int)

    # The number to use for padding is equal to the difference between n_left_out and batch_size*n_gpus.
    pad = batch_size * n_gpus - n_left_out

    return pad.astype(int)


def pad_dataframe(df, n_pad):
    if n_pad:
        df_pad = pd.concat([df.iloc[[-1]]] * n_pad)
        return pd.concat([df, df_pad], ignore_index=True)
    else:
        return df