'''This file is selecting the samples used for the evaluation to be used for all the experiments for cifar10 and GTSRB.'''
import sqlite3

# get the 1000 images samples from the database
conn = sqlite3.connect('Results/db/cifar10_answers.db')
cursor = conn.cursor()
cursor.execute("SELECT Image_path FROM answers")
samples = cursor.fetchall()
conn.close()

# unique samples
samples = list(set(samples))  # Remove duplicates

# save the samples to a file
with open('Results/db/cifar10_1000_samples.txt', 'w') as f:
    for sample in samples:
        f.write("%s\n" % sample[0])

conn.close()

# get the ~900 images samples from the database GTSRB
conn = sqlite3.connect('Results/db/answers.db')
cursor = conn.cursor()
cursor.execute("SELECT Image_path FROM answers WHERE LLM_name = 'gpt-4o'")
samples = cursor.fetchall()
conn.close()

samples = list(set(samples))  # Remove duplicates
num_samples = len(samples)
# save the samples to a file
with open(f'Results/db/gtsrb_{num_samples}_all_sequences_samples.txt', 'w') as f:
    for sample in samples:
        sample = sample[0].replace('all_sequences_training', 'Training_JPEG')
        f.write("%s\n" % sample)

conn.close()