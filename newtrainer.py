# -*- coding: utf-8 -*-

import os
from transformers import (GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

data_folder = r"data"
model_output_dir = "./gpt2-kazakh"

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_output_dir)
tokenizer = GPT2Tokenizer.from_pretrained(model_output_dir)

new_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=os.path.join(data_folder, "tengri_almaty.txt"),
    block_size=128
)

training_args = TrainingArguments(
    output_dir=model_output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    ),
    train_dataset=new_dataset,
)

trainer.train()

model.save_pretrained(model_output_dir, save_format='tf', push_to_hub=False, from_tf=False, ignore_files=None,
                      save_encoding="utf-8")
tokenizer.save_pretrained(model_output_dir, save_encoding="utf-8")
