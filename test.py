class Solo():
    def __init__(self):
        self.half = 1
        self.halfhalf = self.change()

    def change(self):
        self.half = 2
        return 10

sol = Solo()
print sol.half