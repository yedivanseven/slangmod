from ..ml import TrainData, TestData


def log_total_number_of_files(files: list[str]) -> str:
    return f'Found {len(files)} files.'


def log_total_number_of_docs(files: list[str]) -> str:
    return f'Processing {len(files)} documents.'


def log_total_number_of_tokens(sequences: list[list[int]]) -> str:
    n_tokens = sum(len(sequence) for sequence in sequences)
    return f'Converting {n_tokens} tokens to CPU tensor.'


def log_data_sizes(
        train: TrainData,
        test: TestData,
        validation: TestData
) -> str:
    return (f'Created datasets with sizes train={train.n}, '
            f'test={test.n}, and validation={validation.n}.')


def log_validation_metrics(
        loss: float,
        perplexity: float,
        acc: tuple[float, float],
        top_2: tuple[float, float],
        top_5: tuple[float, float],
) -> str:
    return (f'Validation loss: {loss:7.5f} | '
            f'perplexity: {perplexity:6.2f} | '
            f'accuracy: {acc[0]:4.2f}±{acc[1]:4.2f} | '
            f'top-2 acc.: {top_2[0]:4.2f}±{top_2[1]:4.2f} | '
            f'top-5 acc.: {top_5[0]:4.2f}±{top_5[1]:4.2f}')
