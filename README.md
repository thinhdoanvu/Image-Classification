# Image-Classification
Data folder structure
![image](https://github.com/thinhdoanvu/Image-Classification/assets/22977443/b8e40b28-a20a-4211-bc85-9ff821965842)
Train data:
![image](https://github.com/thinhdoanvu/Image-Classification/assets/22977443/848e6099-982f-406b-8de8-7d943799a1cb)
Test data:
![image](https://github.com/thinhdoanvu/Image-Classification/assets/22977443/fbb719e7-70dc-493d-8c5e-c282a85a6a50)
Valid data:
![image](https://github.com/thinhdoanvu/Image-Classification/assets/22977443/fa277613-29b8-461b-9b84-412d3cb19db9)

### Import dependencies
1. Pytorch
2. datasets library
3. Neural network, load and save files
4. Optimizer
5. Dataset, DataLoader
6. ImageFolder
7. transforms
8. ToTensor, Resize
9. numpy

### Definine values
train_data = './data/train'
valid_data = './data/valid'
n_epoch = 1  #at least 100

### Setup CUDA
return cuda or cpu use for training, testing...

### def train_model():
    for batch in train_loader:
        # Set the gradients to zero before starting backpropagation
        # Get a batch
          X: train, y = label
          X, y = X.to(device), y.to(device)
        # Perform a feed-forward pass
        logits = clf(X)  # yhat = predict

        # Compute the batch loss
        loss = loss_fn(logits, y)  # y:label

        # Compute gradient of the loss fn w.r.t the trainable weights
        loss.backward()

        # Update the trainable weights
        optimizer.step()

        # Accumulate the batch loss
        train_loss += loss.item()

    print(f"Epoch:{epoch} train loss is {loss.item()}")
    return train_loss


### def validate_model():
    with torch.no_grad():
        for batch in valid_loader:  # valid_loader

            # Get a batch
            X, y = batch  # X: train, y = label
            X, y = X.to(device), y.to(device)

            # Perform a feed-forward pass
            logits = clf(X)  # yhat = predict

            # Compute the batch loss
            loss = loss_fn(logits, y)  # y:label

            # Accumulate the batch loss
            valid_loss += loss.item()

            # return valid_loss #valid loader
        print(f"Epoch:{epoch} valid loss is {loss.item()}")
        return valid_loss

### main function
if __name__ == "__main__":
    device = setup_cuda()

    # 1. Load the dataset
    from utils.getdataset import Getdataset
    transform = transforms.Compose([Resize((224, 224)), ToTensor()])
    train_dataset = Getdataset(train_data, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataset = Getdataset(valid_data, transform=transform)
    valid_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 2. Create a segmentation model
    import torchvision.models as models
    clf = models.resnet18(pretrained=True).to(device)

    # 3. Specify loss function and optimizer
    optimizer = Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # 4. Training flow
    for epoch in range(n_epoch):  # train for n_epoch
        # 5.1. Train the model over a single epoch
        train_model()

        # 5.2. Validate the model
        validate_model()

        with open('model_state.pt', 'wb') as f:
            save(clf.state_dict(), f)

    # predict
    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(load(f))

    img = Image.open('01.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    print(torch.argmax(clf(img_tensor)))
    
    ...
