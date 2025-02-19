import argparse

def main(args):
    print("hi")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)