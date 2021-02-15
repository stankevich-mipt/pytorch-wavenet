import tensorflow as tf
import numpy as np
import scipy.misc
import threading

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Logger:
    def __init__(self,
                 log_interval=50,
                 validation_interval=200,
                 generate_interval=500,
                 trainer=None,
                 generate_function=None):
        self.trainer = trainer
        self.log_interval = log_interval
        self.validation_interval = validation_interval
        self.generate_interval = generate_interval
        self.accumulated_loss = 0
        self.generate_function = generate_function
        if self.generate_function is not None:
            self.generate_thread = threading.Thread(target=self.generate_function)
            self.generate_function.daemon = True

    def log(self, current_step, current_loss):
        self.accumulated_loss += current_loss
        if current_step % self.log_interval == 0:
            self.log_loss(current_step)
            self.accumulated_loss = 0
        if current_step % self.validation_interval == 0:
            self.validate(current_step)
        if current_step % self.generate_interval == 0:
            self.generate(current_step)

    def log_loss(self, current_step):
        avg_loss = self.accumulated_loss / self.log_interval
        print("loss at step " + str(current_step) + ": " + str(avg_loss))

    def validate(self, current_step):
        avg_loss, avg_accuracy = self.trainer.validate()
        print("validation loss: " + str(avg_loss))
        print("validation accuracy: " + str(avg_accuracy * 100) + "%")

    def generate(self, current_step):
        if self.generate_function is None:
            return

        if self.generate_thread.is_alive():
            print("Last generate is still running, skipping this one")
        else:
            self.generate_thread = threading.Thread(target=self.generate_function,
                                                    args=[current_step])
            self.generate_thread.daemon = True
            self.generate_thread.start()


# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
class TensorboardLogger(Logger):
    def __init__(self,
                 log_interval=50,
                 validation_interval=200,
                 generate_interval=500,
                 trainer=None,
                 generate_function=None,
                 log_dir='logs'):
        super().__init__(log_interval, validation_interval, generate_interval, trainer, generate_function)
        self.writer = tf.summary.create_file_writer(log_dir)

    def log_loss(self, current_step):
        # loss
        avg_loss = self.accumulated_loss / self.log_interval

        with self.writer.as_default():
            tf.summary.scalar('loss', avg_loss, step=current_step)

            # parameter histograms
            for tag, value, in self.trainer.model.named_parameters():
                tag = tag.replace('.', '/')
                tf.summary.histogram(tag, value.data.cpu().numpy(), step=current_step, buckets=200)
                if value.grad is not None:
                    tf.summary.histogram(tag + '/grad', value.data.cpu().numpy(), step=current_step, buckets=200)

    def validate(self, current_step):
        avg_loss, avg_accuracy = self.trainer.validate()

        with self.writer.as_default():
            tf.summary.scalar('validation loss', avg_loss, step=current_step)
            tf.summary.scalar('validation accuracy', avg_accuracy, step=current_step)