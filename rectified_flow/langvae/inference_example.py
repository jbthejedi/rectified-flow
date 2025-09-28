import torch
from random import shuffle

from langvae import LangVAE
from langvae.encoders import SentenceEncoder
from langvae.decoders import SentenceDecoder
from saf_datasets import EntailmentBankDataSet
from langvae.data_conversion.tokenization import TokenizedDataSet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
LATENT_SIZE = 128
MAX_SENT_LEN = 32

# decoder = SentenceDecoder("gpt2", LATENT_SIZE, MAX_SENT_LEN, device=DEVICE, device_map="auto")
# encoder = SentenceEncoder("bert-base-cased", LATENT_SIZE, decoder.tokenizer, caching=True, device=DEVICE)

# model = LangVAE.load_from_hf_hub("neuro-symbolic-ai/eb-langvae-bert-gpt2-example")
# model = LangVAE.load_from_hf_hub("neuro-symbolic-ai/eb-langcvae-bert-base-cased-Mistral-7B-v0.3-srl-l128") # dimension mismatch
# model = LangVAE.load_from_hf_hub("neuro-symbolic-ai/wn_eb-langwae-bert-base-cased-gpt2-l128") # repo not found
# model = LangVAE.load_from_hf_hub("neuro-symbolic-ai/eb-langcvae-bert-base-cased-gpt2-srl-l128")
model = LangVAE.load_from_hf_hub("neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128")

model.eval()

dataset = EntailmentBankDataSet.from_resource("pos+lemma+ctag+dep+srl#expl_only-noreps")

# Shuffling data for training.
shuffle(dataset.data)

eval_size = int(0.05 * len(dataset))# Loads model from HuggingFace Hub.
eval_dataset = TokenizedDataSet(sorted(dataset[-eval_size:], key=lambda x: len(x.surface), reverse=True),
                                model.decoder.tokenizer, model.decoder.max_len, caching=True,
                                cache_persistence=f"eb_eval_tok-bert_gpt2_cache.jsonl")


z, _ = model.encode_z(eval_dataset[:10].data)  # Second element is annotation embedding
dec = model.decode_sentences(z) # Generation will be bad, since model has not been trained enough!

print(dec)