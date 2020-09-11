from mnist import MNIST
import minitorch
import visdom
import numpy

vis = visdom.Visdom()
mndata = MNIST("data/")
images, labels = mndata.load_training()


BACKEND = minitorch.make_tensor_functions(minitorch.FastOps)

BATCH = 16
N = 5000

# Number of classes (10 digits)
C = 10

# Size of images (height and width)
H, W = 28, 28
RATE = 0.01


def RParam(*shape):
    r = 0.1 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return minitorch.matmul(
            x.view(batch, 1, in_size),
            self.weights.value.view(1, in_size, self.out_size),
        ).view(batch, self.out_size) + self.bias.value.view(1, self.out_size)


class Conv2d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kh, kw):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kh, kw)
        self.bias = RParam(out_channels, 1, 1)

    def forward(self, input):
        out = minitorch.conv2d(input, self.weights.value) + self.bias.value
        return out


class Network(minitorch.Module):
    """
    Implement a CNN for MNist classification based on LeNet.

    This model should implement the following procedure:

    1. Apply a convolution with 4 output channels and a 3x3 kernel followed by a ReLU (save to self.mid)
    2. Apply a convolution with 8 output channels and a 3x3 kernel followed by a ReLU (save to self.out)
    3. Apply 2D pooling (either Avg or Max) with 2x2 kernel.
    4. Flatten channels, height, and width. (Should be size BATCHx392)
    5. Apply a Linear to size 64 followed by a ReLU and Dropout with rate 25%
    6. Apply a Linear to size C (number of classes).
    7. Apply a logsoftmax over the class dimension.
    """

    def __init__(self):
        super().__init__()

        # For vis
        self.mid = None
        self.out = None

        # TODO: Implement for Task 4.4.
        raise NotImplementedError('Need to implement for Task 4.4')

    def forward(self, x):
        # TODO: Implement for Task 4.4.
        raise NotImplementedError('Need to implement for Task 4.4')


def make_mnist(start, stop):
    ys = []
    X = []
    for i in range(start, stop):
        y = labels[i]
        vals = [0.0] * 10
        vals[y] = 1.0
        ys.append(vals)
        X.append([[images[i][h * W + w] for w in range(W)] for h in range(H)])
    return X, ys


X, ys = make_mnist(0, N)
val_x, val_ys = make_mnist(10000, 10500)
vis.images(numpy.array(val_x).reshape((len(val_ys), 1, H, W))[:BATCH], win="val_images")


model = Network()
for p in model.parameters():
    p.value.type_(BACKEND)

losses = []
for epoch in range(250):
    total_loss = 0.0
    cur = 0
    cur_y = 0

    model.train()
    for i, j in enumerate(range(0, N, BATCH)):
        if N - j <= BATCH:
            continue
        y = minitorch.tensor_fromlist(ys[j : j + BATCH])
        x = minitorch.tensor_fromlist(X[j : j + BATCH])
        x.requires_grad_(True)
        y.requires_grad_(True)
        y.type_(BACKEND)
        x.type_(BACKEND)

        # Forward
        out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
        prob = (out * y).sum(1)
        loss = -prob.sum()
        loss.view(1).backward()
        total_loss += loss
        losses.append(total_loss)

        # Update
        for p in model.parameters():
            if p.value.grad is not None:
                p.update(p.value - RATE * (p.value.grad / float(BATCH)))

        if i % 50 == 2:
            model.eval()
            # Evaluate on held-out batch
            correct = 0
            y = minitorch.tensor_fromlist(val_ys[:BATCH])
            x = minitorch.tensor_fromlist(val_x[:BATCH])
            out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
            for i in range(BATCH):
                m = -1000
                ind = -1
                for j in range(C):
                    if out[i, j] > m:
                        ind = j
                        m = out[i, j]
                if y[i, ind] == 1.0:
                    correct += 1

            print("Epoch ", epoch, " loss ", total_loss, "correct", correct)

            # Visualize test batch
            for channel in range(4):
                vis.images(
                    -1 * model.mid.to_numpy()[:, channel : channel + 1],
                    win=f"mid_images_{channel}",
                    opts=dict(nrow=4, caption=f"mid_images_{channel}"),
                )
            for channel in range(8):
                vis.images(
                    -1 * model.out.to_numpy()[:, channel : channel + 1],
                    win=f"out_images_{channel}",
                    opts=dict(nrow=4, caption=f"out_images_{channel}"),
                )

            total_loss = 0.0
            model.train()
