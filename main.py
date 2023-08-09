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

    model_save_path = "/content/drive/MyDrive/tensors.pth"
    th.save(model.state_dict(), model_save_path)    


    #few examples
    example_texts = [
        {"messageBody": "What is your return policy?"}, #ham
        {"messageBody": "Hi it's TanyaBot! Click here to see my profile and chat with me for some fun ;) http://chatwithhotbots.com/tanyabot"}, #fake account/bot account spam
        {"messageBody": "FLASH SALE this weekend only! Get 75% off our entire site - huge savings but only while supplies last!"}, #promotional spam
        {"messageBody": "Your Amazon order #8472951 has been delayed. Please click here to reschedule delivery: http://reshipmyorderz.net"}, #phishing link spam
        {"messageBody": "Urgent! Your computer is infected with a virus. Click here to download anti-virus software and remove it now: http://malwarescam.com"}, #malware scam
        {"messageBody": "You've been selected as our lucky winner to receive a free $1000 Walmart gift card! Call 1-800-GIFTCARDS now and provide your credit card number to claim."} #scam offer spam
    ]



    import gradio as gr

    #gradio demo to test out above examples (or any others)
    def load_model():
        model = SpamDetectionModel()
        model.load_state_dict(th.load("/content/drive/MyDrive/tensors.pth"))
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-tiny-finetuned-sms-spam-detection")
        return model, tokenizer

    def predict_spam(text):
        model, _ = load_model()
        model.eval()  # Ensure the model is in evaluation mode

        # Call predict_examples function directly with the input text
        predicted_output = model.predict_examples(text)

        return predicted_output

    iface = gr.Interface(fn=predict_spam, inputs="text", outputs="text")
    iface.launch()

if __name__ == '__main__':
    main()










