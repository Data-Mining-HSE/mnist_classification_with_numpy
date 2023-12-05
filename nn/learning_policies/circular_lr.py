class CircularLR:
    def __init__(self, initial_lr, max_lr, step_size, gamma=0.999):
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.gamma = gamma
        self.cycle = 0
        self.current_lr = initial_lr

    def update_learning_rate(self):
        cycle_position = self.cycle % (2 * self.step_size)
        if cycle_position < self.step_size:
            # Increase learning rate linearly
            self.current_lr = self.initial_lr + (cycle_position / self.step_size) * (self.max_lr - self.initial_lr)
        else:
            # Decrease learning rate exponentially
            self.current_lr = self.max_lr - (cycle_position - self.step_size) / self.step_size * (self.max_lr - self.initial_lr)
            self.current_lr *= self.gamma ** (self.cycle // (2 * self.step_size))

        self.cycle += 1
        return self.current_lr
