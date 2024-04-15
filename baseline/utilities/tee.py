import sys

class Tee(object):
    def __init__(self, name):
        self.file = open(name, 'a')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

    def write(self, data):
        self.file.write(data)
        if self.stderr and hasattr(data, 'encode'):
            self.stderr.write(data)
        elif self.stdout:
            self.stdout.write(data)
        self.file.flush()

    def flush(self):
        self.file.flush()
        if self.stdout:
            self.stdout.flush()
        if self.stderr:
            self.stderr.flush()