import IPython
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

### Utilities


def running_average(arr: np.ndarray, m: int) -> np.ndarray:
    """
    Compute running averages along axis 1 with window size m.
    For the first m-1 columns, average only from the start up to that point.
    
    Parameters:
    - arr: np.ndarray of shape (d, n)
    - m: int, window size

    Returns:
    - np.ndarray of shape (d, n) with running averages
    """
    d, n = arr.shape
    cumsum = np.cumsum(arr, axis=1)
    result = np.zeros((d, n))

    # First m-1 entries: average from the beginning
    for i in range(m):
        result[:, i] = cumsum[:, i] / (i + 1)

    # Entries from m onward: use rolling window average
    result[:, m:] = (cumsum[:, m:] - cumsum[:, :-m]) / m

    return result


def allocate_counts(N, probs):
    """
    Allocates integer counts n_1, ..., n_K such that:
      - sum(n_i) = N
      - n_i ≈ p_i * N
    
    Args:
        N (int): Total count to allocate.
        probs (list or np.array): List of probabilities (must sum to 1).

    Returns:
        list of int: Allocated counts [n1, ..., nK]
    """
    K = len(probs)
    probs = np.array(probs)
    assert np.isclose(probs.sum(), 1.0), "Probabilities must sum to 1"

    # Step 1: Compute ideal (real-valued) allocations
    real_alloc = N * probs

    # Step 2: Take the floor of each
    int_alloc = np.floor(real_alloc).astype(int)

    # Step 3: Distribute the remaining units
    remainder = real_alloc - int_alloc
    num_missing = N - int_alloc.sum()

    # Get indices of largest remainders
    if num_missing > 0:
        add_indices = np.argsort(-remainder)[:num_missing]
        int_alloc[add_indices] += 1

    return int_alloc.tolist()


softmax_temp = 8
batch_size = 64
test_batch_size = 1000
num_classes = 10
epochs = 5

dataset = "MNIST"



# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Data loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 -> 784
])

if dataset == "MNIST":
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

elif dataset == "CFAR10":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform_test)

else:
    raise ValueError("unknown dataset")


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


### create the datasets per class
class_indices = {i: [] for i in range(num_classes)}
for idx, (_, label) in enumerate(train_dataset):
    class_indices[label].append(idx)


# Create a list of 10 DataLoaders, one per digit class
loaders_per_class = []
loaders_iter_per_class = []
for digit in range(num_classes):
    subset = Subset(train_dataset, class_indices[digit])
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    ### creating a list of iterators. Don't want to recreate the iterator every time.
    loaders_per_class.append(loader)
    loaders_iter_per_class.append(iter(loader))



### Create a per class split of the test

test_class_indices = {i: [] for i in range(num_classes)}
for idx, (_, label) in enumerate(test_dataset):
    test_class_indices[label].append(idx)


test_loaders_per_class = []
for digit in range(num_classes):
    subset = Subset(test_dataset, test_class_indices[digit])
    test_loader = DataLoader(subset, batch_size=test_batch_size, shuffle=False)
    ### creating a list of iterators. Don't want to recreate the iterator every time.
    test_loaders_per_class.append(test_loader)





# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

if dataset == "MNIST":
    model_vanilla = MLP().to(device)
    model_loss_selection = MLP().to(device)
elif dataset == "CIFAR10":
    model_vanilla = torchvision.models.resnet18(num_classes=10)
    model_loss_selection = torchvision.models.resnet18(num_classes=10)
else:
    raise ValueError("Unknown dataset")



optimizer_vanilla = optim.Adam(model_vanilla.parameters(), lr=1e-3)
optimizer_loss_selection = optim.Adam(model_loss_selection.parameters(), lr=1e-3)


# Training loop
probabilities_evolution = []
for epoch in range(epochs):
    model_vanilla.train()
    for x, y in train_loader:

        ### Train vanilla
        x, y = x.to(device), y.to(device)
        logits_vanilla = model_vanilla(x)
        loss_vanilla = criterion(logits_vanilla, y)
        optimizer_vanilla.zero_grad()
        loss_vanilla.backward()
        optimizer_vanilla.step()


        ### Train using loss selection

        ### compute losses per class
        loss_stats = []
        next_batches = []



        for class_loader_iter, class_loader,i in zip(loaders_iter_per_class, loaders_per_class, range(num_classes)):

            ### batch to compute stats
            #x_class, y_class = next(iter(class_loader))
            try:
                x_class, y_class = next(class_loader_iter)
                x_class_next, y_class_next = next(class_loader_iter)

            except StopIteration:
                #print("Class iterator ended!!")
                ### It seems like it is triggering the stop iteration all the time - fix this?
                loaders_iter_per_class[i] = iter(class_loader)
                class_loader_iter = loaders_iter_per_class[i]
                x_class, y_class = next(class_loader_iter)
                x_class_next, y_class_next = next(class_loader_iter)

            x_class, y_class = x_class.to(device), y_class.to(device)
            x_class_next, y_class_next = x_class_next.to(device), y_class_next.to(device)

            next_batches.append((x_class_next, y_class_next))

            logits_class = model_loss_selection(x_class)
            loss_class = criterion(logits_class, y_class)
            loss_stats.append( loss_class.item() )

        ### compute the proportion to sample per class
        ### use an exponential softmax rule
        loss_stats = np.array(loss_stats)/np.linalg.norm(loss_stats) ### normalize the loss stats
        #loss_stats = 

        exponentiated_losses =  [np.exp(l*softmax_temp) for l in loss_stats ]
        normalization_factor = sum(exponentiated_losses)
        probabilities = [e/normalization_factor for e in exponentiated_losses]
        probabilities_evolution.append(probabilities)


        count_allocation = allocate_counts(batch_size, probabilities)

        #print("count allocation ", count_allocation)

        ### Create the batch
        filtered_batch = [( x[:a,:], y[:a] ) for ((x,y), a) in zip(next_batches, count_allocation)]
        x_combined = torch.cat([x for (x,y) in filtered_batch ], dim = 0)
        y_combined = torch.cat([y for (x,y) in filtered_batch ], dim = 0)

        #IPython.embed()
        logits_loss_selection = model_loss_selection(x)
        loss_loss_selection = criterion(logits_loss_selection, y)
        optimizer_loss_selection.zero_grad()
        loss_loss_selection.backward()
        optimizer_loss_selection.step()


    print(f"Epoch {epoch+1}/{epochs}, Vanilla Loss: {loss_vanilla.item():.4f}")
    print(f"Epoch {epoch+1}/{epochs}, Loss Selection Loss - not representative: {loss_loss_selection.item():.4f}")

# Evaluate on test set
model_vanilla.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model_vanilla(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"Vanilla Test Accuracy: {100.0 * correct / total:.2f}%")
vanilla_accuracy = 100.0 * correct / total

### Per class test accuracy
accuracy_stats = []
for test_loader_class in test_loaders_per_class: 
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader_class:
            x, y = x.to(device), y.to(device)
            preds = model_vanilla(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    accuracy_stats.append(100.0*correct/ total)

print("Per class accuracy ", accuracy_stats)

# Evaluate on test set
model_loss_selection.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model_loss_selection(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"Loss Selection Test Accuracy: {100.0 * correct / total:.2f}%")
loss_accuracy = 100.0 * correct / total

### Per class test accuracy
loss_accuracy_stats = []
for test_loader_class in test_loaders_per_class: 
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader_class:
            x, y = x.to(device), y.to(device)
            preds = model_loss_selection(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    loss_accuracy_stats.append(100.0*correct/total)

print("Per class accuracy ", loss_accuracy_stats)

labels = [str(i) for i in range(num_classes)] + ["Total"]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width/2, accuracy_stats + [vanilla_accuracy], width, label='Vanilla')
ax.bar(x + width/2, loss_accuracy_stats + [loss_accuracy], width, label='Loss Method')

ax.set_xlabel('Labels')
ax.set_ylabel('Accuracy')
ax.set_title('Histogram of Vanilla and Loss Method {}'.format(dataset))
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(93,100)
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.show()


### plot the probabilities_evolution
average_step = 100
probabilities_evolution = np.array(probabilities_evolution).transpose()
probabilities_evolution = running_average(probabilities_evolution, average_step)

(d,n) = probabilities_evolution.shape
#IPython.embed()

plt.figure(figsize=(10, 6))
for dim in range(d):
    plt.plot(range(n), probabilities_evolution[dim, :], label=f'Label {dim}')


#plt.subplots_adjust(right=0.75)

plt.xlabel('Batch number')
plt.ylabel('Probability')
plt.title('Evolution of class probabilities {}'.format(dataset))
plt.legend()
plt.grid(True)
plt.legend(loc='upper left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.show()




IPython.embed()