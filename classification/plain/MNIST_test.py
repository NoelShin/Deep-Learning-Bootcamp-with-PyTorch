import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
if __name__ == '__main__':
    MODEL = 'MLP'

    # Construct input pipeline
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    dataset = datasets.MNIST(root='./datasets', train=False, transform=transform, download=True)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    # Define model
    if MODEL == 'CNN':
        from MNIST_models import CNN
        model = CNN()
    elif MODEL == 'MLP':
        from MNIST_models import MLP
        model = MLP()
    else:
        raise NotImplementedError("Invalid model type {}. Choose in [CNN, MLP]".format(MODEL))

    # Load the trained model
    state_dict = torch.load('MNIST_model_{}.pt'.format(MODEL)).state_dict()
    model.load_state_dict(state_dict)

    # Test loop
    total_step = 0
    nb_correct_answers = 0
    for i, data in enumerate(data_loader):
        total_step += 1
        input_tensor, label = data[0], data[1]
        input_tensor = input_tensor.view(input_tensor.shape[0], -1) if MODEL == 'MLP' else input_tensor
        classification_results = model(input_tensor)
        nb_correct_answers += torch.eq(classification_results.argmax(), label).item()
    print("Average acc.: {} %.".format(nb_correct_answers / len(dataset) * 100))

