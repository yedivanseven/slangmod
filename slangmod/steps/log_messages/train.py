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
        acc1: float,
        acc2: float,
        acc5: float
) -> str:
    return (f'Loss: {loss:7.5f} | accuracy: {acc1:7.5f} | top-2 accuracy: '
            f'{acc2:7.5f} | top-5 accuracy {acc5:7.5f}')
