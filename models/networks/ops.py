import torch
# tunit model

def queue_data(data, k):
    return torch.cat([data, k], dim=0)

def dequeue_data(data, K=1024):
    if len(data) > K:
        return data[-K:]
    else:
        return data

def initialize_queue(model_k, device, train_loader, feat_size=64):
    queue = torch.zeros((0, feat_size), dtype=torch.float)
    queue = queue.to(device)

    for _, (data, _) in enumerate(train_loader):
        x_k = data[1]
        x_k = x_k.cuda(device)
        outs = model_k(x_k)
        k = outs['cont']
        k = k.detach()
        queue = queue_data(queue, k)
        queue = dequeue_data(queue, K=1024)
        break
    return queue

class MemoryBank(object):
    def __init__(self, device):
        self.device = device
        self.dim = 64
        self.len = 128
        self.queue = torch.zeros((0, self.dim), dtype=torch.float).to(device)

    def init_queue(self, models, train_loader):
        for idx, (_, ref_data, _) in enumerate(train_loader):
            ref_data = ref_data.to(self.device)
            _, out = models(ref_data)
            out = out.detach()
            self.queue = torch.cat([self.queue, out], dim=0)
            if idx+1 >= self.len:
                break
        return self.queue

    def get_queue(self):
        return self.queue

    def update_queue(self, data):
        self.queue = torch.cat([self.queue, data], dim=0)
        if len(self.queue) > self.len:
            self.queue = self.queue[-self.len:]