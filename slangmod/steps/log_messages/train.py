from ...ml import TrainData, TestData


def log_data_sizes(
        train: TrainData,
        test: TestData,
        validation: TestData
) -> str:
    return (f'Created datasets with sizes train={train.n}, '
            f'test={test.n}, and validation={validation.n}.')
