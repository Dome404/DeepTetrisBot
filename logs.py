from keras.callbacks import TensorBoard
from tensorflow.summary import create_file_writer, scalar   

class CustomTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer = create_file_writer(self.log_dir)

    def set_model(self, model):
        pass

    def log(self, step, **stats):
        if stats is None:
            stats = {}
        with self.writer.as_default():
            for name, value in stats.items():
                scalar(name, value, step=step)
            self.writer.flush()
