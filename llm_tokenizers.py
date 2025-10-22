#%%
from transformers import AutoModelForCausalLM, AutoTokenizer
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    dtype="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
#%%
from transformers import pipeline

# Create a pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=200, # Reduced max_new_tokens
    do_sample=False,
    use_cache=False # Disable caching
)
#%%
# @title
# The promt (user input / query)
messages = [
    {"role": "user", "content": "Create a summary for a Youtube video suggestion on how to build an LLM project."}
]

# Generate output
output = generator(messages)
print(output[0]["generated_text"])
#%%
# Tokens and embeddings
prompt = "Write an email to my subscribers how to achieve their goals by constantly upskilling and believing in themselves. <|assistant|>"

# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# Generate the text
generation_output = model.generate(
    input_ids=input_ids,
    max_new_tokens=20,
    use_cache=False # Disable caching to fix the AttributeError
)

# Print the output
print(tokenizer.decode(generation_output[0]))
#%%
input_ids
#%%
for id in input_ids[0]:
  print(tokenizer.decode(id))
#%%
generation_output
#%%
for output in generation_output[0]:
  print(tokenizer.decode(output))
#%%
generation_output
#%%
print(tokenizer.decode(501))
print(tokenizer.decode(567))
print(tokenizer.decode(29895))
print(tokenizer.decode(8873))
print(tokenizer.decode([501,567,29895,8873]))
#%% md
# The tokenization methods are chosen by the model creators. The popular methods are
# 
# *   byte pair encoding  (BPE) -- widely used by GPT models
# *   WordPiece -- used by BERT (bidirectional encoder representations from transformers)
#%%
text = """
English and CAPITALIZATION
🎵 鸟
show_tokens False None elif == >= else: two tabs:"    " Three tabs: "       "
12.0*50=600
"""
#%%
# Comparing trained LLM tokenizers

colors_list = [
    '102;194;165', '252;141;98', '141;160;203',
    '231;138;195', '166;216;84', '255;217;47'
]

def show_tokens(sentence, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_ids = tokenizer(sentence).input_ids
    for idx, t in enumerate(token_ids):
        print(
            f'\x1b[0;30;48;2;{colors_list[idx % len(colors_list)]}m' +
            tokenizer.decode(t) +
            '\x1b[0m',
            end=' '
        )
#%%
show_tokens(text, "bert-base-uncased")
#%%
show_tokens(text, "bert-base-cased")
#%%
show_tokens(text, "gpt2")
#%%
input_ids
#%% md
# gpt2 differs from the tokenization shown in book, possibly updated
#%%
show_tokens(text, "google/flan-t5-small")
#%%
show_tokens(text, "Xenova/gpt-4")
#%%
show_tokens(text, "bigcode/starcoder2-15b")
#%% md
# for the starcoder2 no asccess request needed
#%%
show_tokens(text, "facebook/galactica-1.3b")
#%%
show_tokens(text, "microsoft/Phi-3-mini-4k-instruct")