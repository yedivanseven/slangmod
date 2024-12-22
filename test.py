from slangmod.steps.train import load_data
from memory_profiler import profile

@profile
def run():
    _ = load_data()


if __name__ == '__main__':
    run()

