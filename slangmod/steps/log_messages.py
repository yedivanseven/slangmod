from pathlib import Path
from ..ml import TrainData, TestData


def log_encode_file(file: str) -> str:
    return f'Encoding "{Path(file).name}".'


def log_process_file(file: str) -> str:
    return f'Processing "{Path(file).name}".'


def log_total_number_of_files(files: list[str]) -> str:
    return f'Found {len(files)} file(s).'


def log_total_number_of_docs(files: list[str]) -> str:
    return f'Loaded {len(files)} encoded document(s).'


def log_number_of_tokens(sequences: list[list[int]]) -> str:
    n_tokens = sum(len(sequence) for sequence in sequences)
    return f'Converting {n_tokens} tokens to CPU tensor.'


def log_remaining_number_of_sequences(sequences: list[list[int]]) -> str:
    return f'After filtering, {len(sequences)} sequences remain.'


def log_data_sizes(
        train: TrainData,
        test: TestData,
        validation: TestData
) -> str:
    return (f'Created datasets with sizes train={train.n}, '
            f'test={test.n}, and validation={validation.n}.')


def log_evaluation_metrics(
        loss: float,
        perplexity: float,
        acc: float,
        top_2: float,
        top_5: float,
) -> str:
    return (f'Validation loss: {loss:7.5f} | '
            f'perplexity: {perplexity:6.2f} | '
            f'accuracy: {acc:4.2f} | '
            f'top-2 acc.: {top_2:4.2f} | '
            f'top-5 acc.: {top_5:4.2f}')


def save_evaluation_metrics(
        loss: float,
        perplexity: float,
        acc: float,
        top_2: float,
        top_5: float,
) -> str:
    return ( '[validation]\n'
            f'loss = {loss:7.5f}\n'
            f'perplexity = {perplexity:6.2f}\n'
            f'accuracy = {acc:4.2f}\n'
            f'top_2_acc = {top_2:4.2f}\n'
            f'top_5_acc = {top_5:4.2f}\n')
