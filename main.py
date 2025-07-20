import argparse
from src.data import load_data
from src.models import simple_cnn, medium_cnn, complex_cnn
from src.train import train_model
from src.evaluate import evaluate_model

MODEL_MAP = {
    'simple': simple_cnn,
    'medium': medium_cnn,
    'complex': complex_cnn,
}

def main():
    parser = argparse.ArgumentParser(description="Train CNN for Rock-Paper-Scissors")
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='simple')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--img-size', type=int, nargs=2, default=(64, 64))
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data(args.data_dir, img_size=tuple(args.img_size))

    model_fn = MODEL_MAP[args.model]
    model = model_fn(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])

    train_model(model, X_train, y_train, X_test, y_test,
                epochs=args.epochs, batch_size=args.batch_size)

    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
