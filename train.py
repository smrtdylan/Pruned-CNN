import main
import network
if __name__ == "__main__":
    for epoch in range(11): # train for 10 epochs
        for batch in dataset:
            X,y = batch
            X, y = X.to('cuda'), y.to('cuda')
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            # Apply backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch+1}: loss is {loss.item()}")

    with open('model_state.pt', 'wb') as f:
        save(clf.state_dict(), f)