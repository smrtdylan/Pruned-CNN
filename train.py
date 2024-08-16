import network
from torch import nn,save,load
import torch
from PIL import Image

if __name__ == "__main__":
    for epoch in range(11): # train for 10 epochs
        for batch in network.dataset:
            X,y = batch
            X, y = X.to('cuda'), y.to('cuda')
            yhat = network.clf(X)
            loss = network.loss_fn(yhat, y)

            # Apply backprop
            network.opt.zero_grad()
            loss.backward()
            network.opt.step()

        print(f"Epoch {epoch+1}: loss is {loss.item()}")

    with open('model_state.pt', 'wb') as f:
        network.save(clf.state_dict(), f)