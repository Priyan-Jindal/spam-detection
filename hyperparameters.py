import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

#calculating 95th percentile for max sequence length
data = pd.read_csv('https://raw.githubusercontent.com/PriyanJindal/spamdetection/main/inboxV4.csv')

tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-tiny-finetuned-sms-spam-detection")

# Tokenize all messages and collect their lengths
message_lengths = []
for message in data['messageBody']:
    encoded = tokenizer.encode(message, add_special_tokens=True)
    message_lengths.append(len(encoded))

# Plot a histogram
import matplotlib.pyplot as plt
plt.hist(message_lengths, bins=50, alpha=0.7, color='b')
plt.xlabel('Message Length')
plt.ylabel('Frequency')
plt.title('Distribution of Message Lengths')
plt.show()

# Calculate 95th percentile
percentile_95 = int(np.percentile(message_lengths, 95))

# Alternatively, set a maximum sequence length as a threshold for outliers
max_sequence_length = 200
sequence_length = min(percentile_95, max_sequence_length)

print("Chosen Sequence Length:", sequence_length)
