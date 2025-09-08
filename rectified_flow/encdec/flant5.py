# text_backbones.py (for example)
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

class FlanT5TextEncoderDecoder(nn.Module):
  """
    Frozen Flan-T5 encoder/decoder adapter for joint RF.

    - encode(input_ids, attention_mask) -> encoder hidden states H ∈ [B, L, d]
    - decode(H_hat, attention_mask) -> list[str] captions
    """
  def __init__(self, model_name: str = "google/flan-t5-small", device: str = "cuda"):
    super().__init__()
    self.device = torch.device(device)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    self.model.to(self.device)
    self.model.eval()
    for p in self.model.parameters():
      p.requires_grad_(False)

    # useful for wiring UNet cross-attn dims
    self.hidden_size = self.model.config.d_model  # e.g. 512 for flan-t5-small


  @torch.no_grad()
  def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """
        input_ids:    [B, L] (already from same tokenizer)
        attention_mask: [B, L]
        returns H:    [B, L, d_model]
        """
    enc_out = self.model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    )
    H = enc_out.last_hidden_state  # [B, L, d]
    return H


  @torch.no_grad()
  def decode(
      self,
      encoder_hidden_states: torch.Tensor,
      attention_mask: torch.Tensor = None,
      max_new_tokens: int = 32,
      num_beams: int = 1,
  ):
    """
        encoder_hidden_states: [B, L, d] (e.g. RF-denoised Ĥ)
        attention_mask: [B, L] (optional but recommended)
        returns: list[str] decoded texts
        """
    encoder_hidden_states = encoder_hidden_states.to(self.device)
    if attention_mask is not None:
      attention_mask = attention_mask.to(self.device)

    encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

    gen_ids = self.model.generate(
        encoder_outputs=encoder_outputs,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    return texts
