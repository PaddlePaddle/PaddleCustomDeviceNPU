# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
from paddle.optimizer import SGD

paddle.set_device("intel_hpu")

BATCH_SIZE = 64


class MnistDataset(paddle.vision.datasets.MNIST):
    def __init__(self, mode, return_label=True):
        super(MnistDataset, self).__init__(mode=mode)
        self.return_label = return_label

    def __getitem__(self, idx):
        img = np.reshape(self.images[idx], [1, 28, 28])
        img = img / 255.0 * 2.0 - 1.0
        if self.return_label:
            return img, np.array(self.labels[idx]).astype("int")
        return (img,)

    def __len__(self):
        return len(self.images)


train_reader = paddle.io.DataLoader(
    MnistDataset(mode="train"), batch_size=BATCH_SIZE, drop_last=True
)
test_reader = paddle.io.DataLoader(
    MnistDataset(mode="test"), batch_size=BATCH_SIZE, drop_last=True
)


class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()
        self.shape = 1 * 28 * 28
        self.size = 10
        self.output_weight = self.create_parameter([self.shape, self.size])
        self.accuracy = paddle.metric.Accuracy()

    def forward(self, inputs, label=None):
        x = paddle.reshape(inputs, shape=[-1, self.shape])
        x = paddle.matmul(x, self.output_weight)
        x = paddle.nn.functional.softmax(x)
        if label is not None:
            self.accuracy.reset()
            correct = self.accuracy.compute(x, label)
            self.accuracy.update(correct)
            acc = self.accuracy.accumulate()
            return x, acc
        else:
            return x


mnist = MNIST()
sgd = SGD(learning_rate=0.01, parameters=mnist.parameters())

epoch_num = 1
for epoch in range(epoch_num):
    for batch_id, data in enumerate(train_reader()):
        img = data[0]
        label = data[1]

        pred, acc = mnist(img, label)
        avg_loss = paddle.nn.functional.cross_entropy(pred, label)
        avg_loss.backward()
        sgd.step()
        sgd.clear_grad()

        if batch_id % 100 == 0:
            print(
                "Epoch {} step {}, Loss = {:}, Accuracy = {:}".format(
                    epoch, batch_id, avg_loss.numpy(), acc
                )
            )
model_dict = mnist.state_dict()
paddle.save(model_dict, "mnist.pdparams")
