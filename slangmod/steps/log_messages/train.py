from ...ml import TrainData, TestData


def log_data_sizes(
        train: TrainData,
        test: TestData,
        validation: TestData
) -> str:
    return (f'Created datasets with sizes train={train.n}, '
            f'test={test.n}, and validation={validation.n}.')


def log_validation_metrics(
        loss: float,
        acc: float,
        top_2: float,
        top_5: float
) -> str:
    return (f'Validation loss: {loss:7.5f} | accuracy: {acc:7.5f} | '
            f'top-2 acc.: {top_2:7.5f} | top-5 acc.: {top_5:7.5f}')
