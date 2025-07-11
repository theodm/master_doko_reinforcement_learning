from tensorboardX import SummaryWriter

class TensorboardController:
    writer: SummaryWriter

    def __init__(self, path: str):
        self.writer = SummaryWriter(path)
        pass

    def scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)
        pass