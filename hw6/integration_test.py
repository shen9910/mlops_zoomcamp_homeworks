import batch
from datetime import datetime
import pandas as pd


S3_ENDPOINT_URL = 'http://localhost:4566'

options = {
    'client_kwargs': {
        'endpoint_url': S3_ENDPOINT_URL
            }
        }

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


def test_save_file():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
    ]

    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    df_input = pd.DataFrame(data, columns=columns)

    input_file = batch.get_input_path(2021, 1)

    df_input.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

    assert 1 == 1


def q6_test():
    batch.main(2021, 1)
    saved_file = batch.get_output_path(2021, 1)
    print(saved_file)
    df = pd.read_parquet(saved_file, storage_options=options)
    sum_duration = df['predicted_duration'].sum()
    print(f'Sum of predicted duration {sum_duration}')
    assert 1 == 1


if __name__ == "__main__":
    test_save_file()
    q6_test()


