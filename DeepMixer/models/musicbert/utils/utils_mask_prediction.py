from itertools import chain
from enum import Enum
import torch

class PRED_MODE(Enum):
    VANILLA = 1
    OCTUPLE_MODE = 2
    MULTI_OCTUPLE_MODE = 3

class PRED_STRAT(Enum):
    TOP = 1  #Always takes the top prediction
    TEMPERATURE = 2 #Temperature based on thermodynamics higher value allows lower energy (prob) states to be encountered more 
    TOP_K = 3  #Takes top-k results, performs softmax over then and uniformly samples based on the resultant probabilities. Will also utilize temperature. 


def top_k_top_p(logits_batch, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """

    logits_batch = logits_batch.clone()

    # We have to convert logits_batch to correct dimension when batch size is 1 
    if(logits_batch.dim() == 1):
      logits_batch = logits_batch.unsqueeze(0)

    assert logits_batch.dim() == 2  # check if logits are in correct dimensionality -> (batch, seq_length)
    
    # iterate through batch size 
    for index, logits in enumerate(logits_batch):
      top_k = min(top_k, logits.size(-1))  # Safety check
      if top_k > 0:
          # Remove all tokens with a probability less than the last token of the top-k
          indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
          logits[indices_to_remove] = filter_value

      if top_p > 0.0:
          sorted_logits, sorted_indices = torch.sort(logits, descending=True)
          cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

          # Remove tokens with cumulative probability above the threshold
          sorted_indices_to_remove = cumulative_probs > top_p
          # Shift the indices to the right to keep also the first token above the threshold
          sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
          sorted_indices_to_remove[..., 0] = 0

          indices_to_remove = sorted_indices[sorted_indices_to_remove]
          logits[indices_to_remove] = filter_value
    return logits_batch

#Helper function for OCTUPLE_MODE and MULTI_OCTUPLE_MODE
# https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
def split_multi_oct(list_a, chunk_size):
  for i in range(0, len(list_a), chunk_size):
    yield list(chain(*list_a[i:i + chunk_size])) 

#Predict missing masks in sequence from left to right

'''
NOTE: Predicts ONLY the masked octuples provided in `masked_octuples` if `prediction_mode` is NOT Vanilla
Else if prediction_mode is Vanilla, it predicts all the masks in the input `octuplemidi_token_encoding`
'''

#octuplemidi_token_encoding: of the format torch.Tensor([0,0,0,0,0,0,0,0, ..........., 2,2,2,2,2,2,2,2]), where 0 is label_dict.bos_idx & 2 is label_dict.eos_idx
#prediction_mode: decides the speed of the mask prediction
#mask_attribs: decides which of the (`bar`, `position`, `instrument`, `pitch`, `duration`, `velocity`, `timesig` , `tempo`) are masked

#masked_octuples: List of the octuple indices in `octuplemidi_token_encoding` that are masked, note that the element indices for `bar` field of these elements would be (mask_octuple_idxs * 8)


def predict_all_masks(roberta_model, roberta_label_dict, octuplemidi_token_encoding:torch.Tensor, masked_octuples:list = None,
                      prediction_mode:PRED_MODE = PRED_MODE.VANILLA, mask_attribs:list = [3,4,5] ,num_multi_octuples:int = None,
                      prediction_strategy:PRED_STRAT = PRED_STRAT.TOP, temperature = 1.0, top_k=30, top_p=0.6,
                      verbose = False):
  mask_idx = 1236
  octuplemidi_token_encoding = octuplemidi_token_encoding.clone()

  try:
    assert octuplemidi_token_encoding.dim() == 1
  except:
    raise Exception('Please input single dimensional octuple sequence')

  try:
    bos_idx = roberta_label_dict.bos_index
    eos_idx = roberta_label_dict.eos_index
    tens_type = octuplemidi_token_encoding.dtype
    assert torch.equal(octuplemidi_token_encoding[:8], torch.Tensor([bos_idx]*8).type(tens_type)) and \
      torch.equal(octuplemidi_token_encoding[-8:], torch.Tensor([eos_idx]*8).type(tens_type))
  except:
    print('Start:', octuplemidi_token_encoding[:8] )
    print(torch.Tensor([bos_idx]*8))
    print('End:', octuplemidi_token_encoding[-8:])
    print(torch.Tensor([bos_idx]*8))
    raise Exception('`octuplemidi_token_encoding` either does not have 8 <s> tokens or 8 </s> tokens at beginning and end')

  #---------------------------------------------------------------
  #Altering input mask list based on `prediction_mode`
  #---------------------------------------------------------------

  mask_indices = None

  # If `masked_octuples` not provided, then the `prediction_mode` MUST be Vanilla
  if masked_octuples == None:
    try:  
      assert prediction_mode == PRED_MODE.VANILLA
    except:
      #Since the current faster implementations involves the premise that in all the masked notes, same fields of each octuple is masked,
      #For example, we are not considering that in the sequence one octuple has just `duration` masked and another has just `pitch` masked
      raise Exception("Error: Please choose `prediction_mode` as Vanilla since `masked_octuples` is not provided, to use faster modes provide `mask_indices`")

    mask_indices = [i for i, x in enumerate(octuplemidi_token_encoding.tolist()) if x == mask_idx]

  elif prediction_mode == PRED_MODE.VANILLA:

    print('Warning: Ignoring `masked_octuples`, `mask_attribs` & `num_multi_octuples` as `prediction_mode` is set as Vanilla')
    mask_indices = [i for i, x in enumerate(octuplemidi_token_encoding.tolist()) if x == mask_idx]
    
  elif prediction_mode == PRED_MODE.OCTUPLE_MODE:

    if num_multi_octuples is not None:
      print('Warning: Ignoring `num_multi_octuples` as `prediction_mode` is set as Octuple mode (not Multi-octuple mode)')

    mask_indices = [ [x*8 + y for y in mask_attribs] for x in masked_octuples]

  elif prediction_mode == PRED_MODE.MULTI_OCTUPLE_MODE:

    try:
      assert num_multi_octuples is not None
    except:
      raise Exception("Please provide argument `num_multi_octuples` for Multi-octuple mode")

    try:
      assert num_multi_octuples <= len(masked_octuples)
    except:
      raise Exception("`num_multi_octuples` should be less than number of masked octuples")

    mask_indices = [ [x*8 + y for y in mask_attribs] for x in masked_octuples]
    mask_indices = list(split_multi_oct(mask_indices, num_multi_octuples))

  else:
    raise Exception("Invalid `prediction_mode`")


  try:
    assert len(mask_indices) > 0
  except AssertionError:
    raise Exception('Please input sentence tokens with at least one mask token')

  try:
    assert all( torch.all(octuplemidi_token_encoding[octuple_midi_mask_elem] == mask_idx) for octuple_midi_mask_elem in mask_indices )
  except:
    print([octuplemidi_token_encoding[octuple_midi_mask_elem] == mask_idx for octuple_midi_mask_elem in mask_indices])
    raise Exception('Fatal error: At least one element of `mask_indices` is not <mask> (1236)')
  
  #--------------------------------------------------------------
  #Inputting masked indices to model using `prediction_strategy`
  #--------------------------------------------------------------

  if prediction_mode == PRED_MODE.VANILLA:
    pass

  elif prediction_mode == PRED_MODE.OCTUPLE_MODE:
    
    #Checking if `mask_attribs` is fine
    try:
      mask_attribs_len = len(mask_attribs)
      assert mask_attribs_len > 0 and \
      len(set(mask_attribs)) == mask_attribs_len and \
      all( (x >= 0 and x < 8) for x in mask_attribs)
    except:
      raise Exception("`mask_attribs` not appropriate")


  elif prediction_mode == PRED_MODE.MULTI_OCTUPLE_MODE:

    #Checking if `mask_attribs` is fine
    try:
      mask_attribs_len = len(mask_attribs)
      assert mask_attribs_len > 0 and \
      len(set(mask_attribs)) == mask_attribs_len and \
      all( (x >= 0 and x < 8) for x in mask_attribs)
    except:
      raise Exception("`mask_attribs` not appropriate")

    #Checking if `num_multi_octuples` is fine
    try:
      assert num_multi_octuples is not None and \
      num_multi_octuples > 0 and \
      isinstance(num_multi_octuples, int)
    except:
      raise Exception("`num_multi_octuples` should be appropriate positive integer")

  else:
     raise Exception('Fatal error: Invalid `prediction_mode`')

  print(f'Final mask indices list sent to model: {mask_indices}')
  print(f'Out of total input size: {len(octuplemidi_token_encoding.tolist())}')

  
  #Let's take an example where `mask_attribs` = [3,4,5] and at least 

  # `final_mask_indices` is of form [3,4,5,11,12,13....] in `Vanilla` prediction mode
  # `final_mask_indices` is of form [[3,4,5],[11,12,13]......] in `Octuple` prediction mode
  # `final_mask_indices` is of form [[3,4,5,11,12,13]......] in `Multi Octuple` prediction mode, in this case num_multi_octuples = 2

  filter_value = -float('Inf')
  octuplemidi_token_encoding_device = octuplemidi_token_encoding.device

  # Finally predicting masks based on `prediction_strategy`
  if prediction_mode == PRED_MODE.VANILLA or \
      prediction_mode == PRED_MODE.OCTUPLE_MODE or \
      prediction_mode == PRED_MODE.MULTI_OCTUPLE_MODE:
    
    # Greedy Sampling
    if prediction_strategy == PRED_STRAT.TOP:
      for mask_idx_batch in tqdm(mask_indices):
          input = octuplemidi_token_encoding.unsqueeze(0).cuda()
          with torch.no_grad():
            
            # extr_features shape -> [1, 8016, 1237] 
            # Here 1 is the batch size, 8016 is embedding dimension and 1237 is the vocab size 
            extr_features, _ = roberta_model.model.extract_features(input)
            
            # filter the mask indices from the extracted feature tokens (1000 octuples) 
            logits = extr_features[0, mask_idx_batch ]
            
            probs = torch.softmax(logits, dim = 0)
            
            if probs.dim() == 1:
              probs = probs.unsqueeze(0)
    
            top_preds = torch.argmax(probs, dim = 1)
          octuplemidi_token_encoding_type = octuplemidi_token_encoding.dtype

          octuplemidi_token_encoding[mask_idx_batch] = top_preds.\
                                                      type(octuplemidi_token_encoding_type). \
                                                      to(octuplemidi_token_encoding_device)
      
    # Temperature only Sampling 
    elif prediction_strategy == PRED_STRAT.TEMPERATURE:
       for mask_idx_batch in tqdm(mask_indices):
          input = octuplemidi_token_encoding.unsqueeze(0).cuda()
          with torch.no_grad():
            
            # extr_features shape -> [1, 8016, 1237] 
            # Here 1 is the batch size, 8016 is embedding dimension and 1237 is the vocab size 
            extr_features, _ = roberta_model.model.extract_features(input)
            # filter the mask indices from the extracted feature tokens (1000 octuples) 
            logits = extr_features[0, mask_idx_batch ]
            if temperature != 1. : logits = logits/temperature

            probs = torch.softmax(logits, dim = -1)
            
            if probs.dim() == 1:
              probs = probs.unsqueeze(0)
            
            # We sample from a multinomial distribution 
            top_preds = torch.multinomial(probs, 1)
            top_preds = top_preds.reshape(-1)
          
          octuplemidi_token_encoding_type = octuplemidi_token_encoding.dtype

          octuplemidi_token_encoding[mask_idx_batch] = top_preds.\
                                                      type(octuplemidi_token_encoding_type). \
                                                      to(octuplemidi_token_encoding_device)

    elif prediction_strategy == PRED_STRAT.TOP_K:
       for mask_idx_batch in tqdm(mask_indices):
          input = octuplemidi_token_encoding.unsqueeze(0).cuda()
          with torch.no_grad():
            
            # extr_features shape -> [1, 8016, 1237] 
            # Here 1 is the batch size, 8016 is embedding dimension and 1237 is the vocab size 
            extr_features, _ = roberta_model.model.extract_features(input)
            # filter the mask indices from the extracted feature tokens (1000 octuples) 
            logits = extr_features[0, mask_idx_batch ]

            if temperature != 1. : logits = logits/temperature
            
            logits = top_k_top_p(logits, top_k=top_k, top_p=top_p, filter_value=filter_value)

            probs = torch.softmax(logits, dim = -1)
            if probs.dim() == 1:
              probs = probs.unsqueeze(0)

            # We sample from a multinomial distribution
            top_preds = torch.multinomial(probs, 1)
            top_preds = top_preds.reshape(-1)
          
          octuplemidi_token_encoding_type = octuplemidi_token_encoding.dtype
          octuplemidi_token_encoding[mask_idx_batch] = top_preds.\
                                                      type(octuplemidi_token_encoding_type). \
                                                      to(octuplemidi_token_encoding_device)

    else:
      raise Exception('Fatal error: Invalid `prediction_strategy`')


          
  try:
    assert not any( torch.all(octuplemidi_token_encoding[octuple_midi_mask_elem] == mask_idx) for octuple_midi_mask_elem in mask_indices )
  except:
    print([octuplemidi_token_encoding[octuple_midi_mask_elem] == mask_idx for octuple_midi_mask_elem in mask_indices])
    raise Exception('Fatal error: The prediction has at least one element of `mask_indices` as <mask>')

  return octuplemidi_token_encoding