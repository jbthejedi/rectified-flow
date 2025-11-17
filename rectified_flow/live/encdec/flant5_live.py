import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.models import AutoTokenizer, AutoModelForSeq2SeqLM


class FlanT5TextEncoderDecoder(nn.Module):

  def __init__(self, model_name: str = "google/flan-t5-small", device=None):
    """
    Setup tokenizer and model
    """
    super().__init__()
    self.device = device
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    for p in self.model.parameters():
      p.requires_grad_(False)
    
    self.hidden_size = self.model.config.d_model

  @torch.no_grad()
  def encode(self, input_ids, attn_mask):
    """Encode input ids"""
    return self.model.encoder(input_ids=input_ids, attention_mask=attn_mask,
                             return_dict=True).last_hidden_state

  @torch.no_grad()
  def decode(self, encoder_hidden_states, attention_mask, max_new_tokens, num_beams):
    """Generate input_ids and Batch decode into text"""
    encoder_hidden_states = encoder_hidden_states.to(self.device)
    if attention_mask is not None:
      attention_mask = attention_mask.to(self.device)
    
    encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
    gen_ids = self.model.generate(encoder_outputs=encoder_outputs,
                                  attention_mask=attention_mask,
                                  max_new_tokens=max_new_tokens,
                                  num_beams=num_beams)
    return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
