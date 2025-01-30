import argparse
import setup
import train

def main():
    parser = argparse.ArgumentParser(description='WVCNN DSE')

    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--download-dataset', action='store_true', default=False)
    parser.add_argument('--repo-source', type=str, default='/home/john/9_sandbox/jphdsource/')

    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-2)

    args = parser.parse_args()

    (device, data, label) = setup.setup(args)
    train.train(args, device, data, label)

if __name__ == '__main__':
    main()