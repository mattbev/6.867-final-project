import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run a deep FL pipeline with attacks and defenses.")
    
    parser.add_argument('-n', '--n_epochs', const=1, action='store_const', help="the number of epochs to train for.")
    parser.add_argument()

    args = parser.parse_args()