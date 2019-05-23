import torch
from torch import nn
from torch.nn import functional as F
from deepclustering.utils import simplex


class neural_network_model(nn.Module):
    n_nodes_hl1 = 12
    n_nodes_hl2 = 24
    n_nodes_hl3 = 24
    n_nodes_hl4 = 24
    n_nodes_hl5 = 10
    n_nodes_hl6 = 3
    n_nodes_hl7 = 2
    n_classes = 2

    def __init__(self, ):
        super().__init__()
        self.hidden1 = nn.Linear(12, self.n_nodes_hl1)
        self.hidden2 = nn.Linear(self.n_nodes_hl1, self.n_nodes_hl2)
        self.hidden3 = nn.Linear(self.n_nodes_hl2, self.n_nodes_hl3)
        self.hidden4 = nn.Linear(self.n_nodes_hl3, self.n_nodes_hl4)
        self.hidden5 = nn.Linear(self.n_nodes_hl4, self.n_nodes_hl5)
        self.hidden6 = nn.Linear(self.n_nodes_hl5, self.n_nodes_hl6)
        self.hidden7 = nn.Linear(self.n_nodes_hl6, self.n_nodes_hl7)
        self.output_layer = nn.Linear(self.n_nodes_hl7, self.n_classes)
        self.__init_weights__()

    def forward(self, input):
        output1 = torch.tanh(self.hidden1(input))
        output2 = torch.tanh(self.hidden2(output1))
        output3 = torch.tanh(self.hidden3(output2))
        output4 = torch.tanh(self.hidden4(output3))
        output5 = torch.tanh(self.hidden5(output4))
        output6 = torch.tanh(self.hidden6(output5))
        output7 = torch.tanh(self.hidden7(output6))

        output8 = F.softmax(self.output_layer(output7), 1)
        assert simplex(output8)
        return output8, [output1, output2, output3, output4, output5, output6, output7, output8]

    def __init_weights__(self):
        for k, m in self._modules.items():
            if isinstance(m, nn.Linear):
                self.truncated_normal_(m.weight, std=1.0 / m.out_features)
                torch.nn.init.zeros_(m.bias)

    @staticmethod
    def truncated_normal_(tensor, mean=0, std=1):
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)


if __name__ == '__main__':
    net = neural_network_model()

"""
def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.truncated_normal(
        [12, n_nodes_hl1], mean=0.0, stddev=1.0 / np.sqrt(float(n_nodes_hl1)))),
        'biases': tf.Variable(tf.constant(0.0, shape=[n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.truncated_normal(
        [n_nodes_hl1, n_nodes_hl2], mean=0.0, stddev=1.0 / np.sqrt(float(n_nodes_hl2)))),
        'biases': tf.Variable(tf.constant(0.0, shape=[n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.truncated_normal(
        [n_nodes_hl2, n_nodes_hl3], mean=0.0, stddev=1.0 / np.sqrt(float(n_nodes_hl3)))),
        'biases': tf.Variable(tf.constant(0.0, shape=[n_nodes_hl3]))}

    hidden_4_layer = {'weights': tf.Variable(tf.truncated_normal(
        [n_nodes_hl3, n_nodes_hl4], mean=0.0, stddev=1.0 / np.sqrt(float(n_nodes_hl4)))),
        'biases': tf.Variable(tf.constant(0.0, shape=[n_nodes_hl4]))}

    hidden_5_layer = {'weights': tf.Variable(tf.truncated_normal(
        [n_nodes_hl4, n_nodes_hl5], mean=0.0, stddev=1.0 / np.sqrt(float(n_nodes_hl5)))),
        'biases': tf.Variable(tf.constant(0.0, shape=[n_nodes_hl5]))}

    hidden_6_layer = {'weights': tf.Variable(tf.truncated_normal(
        [n_nodes_hl5, n_nodes_hl6], mean=0.0, stddev=1.0 / np.sqrt(float(n_nodes_hl6)))),
        'biases': tf.Variable(tf.constant(0.0, shape=[n_nodes_hl6]))}

    hidden_7_layer = {'weights': tf.Variable(tf.truncated_normal(
        [n_nodes_hl6, n_nodes_hl7], mean=0.0, stddev=1.0 / np.sqrt(float(n_nodes_hl7)))),
        'biases': tf.Variable(tf.constant(0.0, shape=[n_nodes_hl7]))}

    output_layer = {'weights': tf.Variable(tf.truncated_normal(
        [n_nodes_hl7, n_classes], mean=0.0, stddev=1.0 / np.sqrt(float(n_classes)))),
        'biases': tf.Variable(tf.constant(0.0, shape=[n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.tanh(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.tanh(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.tanh(l3)

    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.tanh(l4)

    l5 = tf.add(tf.matmul(l4, hidden_5_layer['weights']), hidden_5_layer['biases'])
    l5 = tf.nn.tanh(l5)

    l6 = tf.add(tf.matmul(l5, hidden_6_layer['weights']), hidden_6_layer['biases'])
    l6 = tf.nn.tanh(l6)

    l7 = tf.add(tf.matmul(l6, hidden_7_layer['weights']), hidden_7_layer['biases'])
    l7 = tf.nn.tanh(l7)
    output = tf.matmul(l7, output_layer['weights']) + output_layer['biases']
    output = tf.nn.softmax(output)
    layers = [l1, l2, l3, l4, l5, l6, l7, output]
    return layers
"""
