import pytorch_lightning as pl
from spamdetectionmodel import SpamDetectionModel  # Import the SpamDetectionModel class
from torch.utils.tensorboard import TensorBoardLogger  # Import TensorBoardLogger
import os
import shutil

def main():
    model = SpamDetectionModel()
    import os
    import shutil

    # clear logs
    # logs_dir = 'logs/'
    # shutil.rmtree(logs_dir)
    # os.makedirs(logs_dir)

    trainer = pl.Trainer(
        default_root_dir='logs',
        max_epochs=model.epochs,
        logger = TensorBoardLogger(save_dir="logs/", name="spam_detection")
    )
    trainer.fit(model)

    #print a few examples
    example_texts = [
        {"messageBody": "Congratulations! You've won a prize."}, #spam
        {"messageBody": "What is your return policy?"}, #ham
    ]
    model.predict_examples(example_texts)

    %load_ext tensorboard
    %tensorboard --logdir logs

if __name__ == '__main__':
    main()










