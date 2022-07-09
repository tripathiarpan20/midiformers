#Returns dictionary with keys = token_ids & values = token strings
def reverse_label_dict(label_dict: fairseq.data.dictionary.Dictionary):
  return {v: k for k, v in label_dict.indices.items()}

def decode_w_label_dict(label_dict: fairseq.data.dictionary.Dictionary, octuple_midi_enc:torch.Tensor,
                        skip_masked_tokens = False):
  octuple_midi_enc_copy = octuple_midi_enc.clone().tolist()
  seq = []
  rev_inv_map = reverse_label_dict(label_dict)
  for token in octuple_midi_enc_copy:
    seq.append(rev_inv_map[token])

  seq_str = " ".join(seq)

  if skip_masked_tokens:
    seq = seq_str.split()
    masked_oct_idxs = set([(idx - idx%8) for idx, elem in enumerate(seq) if elem == '<mask>'])

    #Deleting Octuples with any <mask> element until none remains
    try:
      while(True):
        masked_oct_idx = seq.index('<mask>')
        masked_oct_idx = masked_oct_idx - masked_oct_idx%8
        del seq[masked_oct_idx: masked_oct_idx+8]
    except ValueError: #Error: substring not found
      pass

    seq_str = " ".join(seq)

  del octuple_midi_enc_copy
  return seq_str