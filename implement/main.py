from session import JAGH
import argparse
import config


def main():
    parser = argparse.ArgumentParser(description='JAGH')
    parser.add_argument('--Dataset', default='MIRFlickr', help='hash bit', type=str)
    parser.add_argument('--BIT', default=128, help='hash bit', type=int)
    parser.add_argument('--gamma', default=0.45, help='mse', type=float)
    parser.add_argument('--pho', default=0.25, help='pho3', type=float)
    parser.add_argument('--alpha', default=0.5, help='alpha', type=float)
    parser.add_argument('--beta', default=0.1, help='beta', type=float)
    parser.add_argument('--lamada', default=0.4, help='lamada', type=float)
    parser.add_argument('--mu', default=1.5, help='mu', type=float)

    args = parser.parse_args()

    config.gamma = args.gamma
    config.pho = args.pho
    config.CODE_LEN = args.BIT
    config.alpha = args.alpha
    config.beta = args.beta
    config.lamada = args.lamada
    config.mu = args.mu

    Model = JAGH()

    if config.TEST == True:
        Model.load_checkpoints()
        Model.eval()

    else:
        for epoch in range(config.NUM_EPOCH):
            Model.train(epoch)
            if (epoch + 1) % config.EVAL_INTERVAL == 0:
                Model.eval()
            # save the model
            if epoch + 1 == config.NUM_EPOCH:
                Model.save_checkpoints()


if __name__ == '__main__':
    main()