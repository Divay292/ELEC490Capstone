# Import necessary libraries
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

argParser = argparse.ArgumentParser()
# training options
argParser.add_argument('-e', type=int, help='Epochs')
argParser.add_argument('-b', type=int, help='Batch Size')
argParser.add_argument('-m', type=str, help='Mode')
argParser.add_argument('-s', type=str, help='Weight File')
argParser.add_argument('-cuda', metavar='cuda', type=str, help='[y/N]')
args = argParser.parse_args()
lr = 1e-3


def train(n_epochs, optimizer, loss_fn, training_set,
          x_train, y_train, model_train, scheduler, device):
    losses_train = []
    print('training...')
    for epoch in range(1, n_epochs+1):
        model_train.train()
        loss_train = 0.0
        print('epochs ', epoch)
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.b, shuffle=True)
        for batch, (imgs, coords) in enumerate(train_loader):

            optimizer.zero_grad()

            loss = loss_fn(outputs, coords)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        losses_train += [loss_train / len(train_loader)]
        print('Loss: ', losses_train[epoch - 1])
        torch.save(model_train.state_dict(), args.s)

        scheduler.step()
    plt.figure(2, figsize=(12, 7))
    plt.clf()
    plt.plot(losses_train, label='train')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(loc=1)
    plt.show()

    return model_train


def test(x_test, y_test, model_test, device):

    print('testing...')
    test_loader = torch.utils.data.DataLoader(x_test, y_test, shuffle=True)
    for batch, (imgs, coords) in enumerate(test_loader):
        model_test.eval()
        with torch.no_grad():
            model_test.load_state_dict(torch.load(args.s))


def main():
    df = pd.read_csv('your_dataset.csv')

    # Data preprocessing
    x = df[['Age', 'Occupation', 'Stress Level', 'BMI Category', 'Heart rate (bpm)',
            'Daily Steps', 'Time in bed (seconds)', 'Time asleep (seconds)',
            'Snore time']]  # Features
    y = df['Sleep Quality']  # Target variable

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    device = torch.device('cuda' if args.cuda == 'y' or args.cuda == 'Y' else 'cpu')
    resnet = models.resnet18()

    net = resnet.to(device)
    loss_fn = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.e)
    if args.m == 'train':
        print('Training')
        train(n_epochs=args.e, optimizer=optimizer,
              loss_fn=loss_fn, x_train=x_train, y_train=y_train, model_train=net, scheduler=scheduler,
              device=device)
    else:
        print('Testing')
        test(x_test=x_test, y_test=y_test, model_test=net, device=device)

    # Train the model
    trained_model = train(x_train, y_train)

    # Make predictions on the test set
    y_pred = trained_model.predict(x_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    plt.scatter(x_train, y_test, color='black', label='Actual')
    plt.plot(x_test, y_pred, color='blue', linewidth=3, label='Predicted')
    plt.xlabel('Sleep Features')
    plt.ylabel('Sleep Quality')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

