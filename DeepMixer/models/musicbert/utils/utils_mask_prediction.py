from itertools import chain
from enum import Enum
import torch


# Storing string encodings for utility functions
BAR_START = "<0-0>"
BAR_END = "<0-255>"

POS_START = "<1-0>"
POS_END = "<1-127>"

INS_START = "<2-0>"
INS_END = "<2-127>"

PITCH_START = "<3-0>"
PITCH_END = "<3-255>"

DUR_START = "<4-0>"
DUR_END = "<4-127>"

VEL_START = "<5-0>"
VEL_END = "<5-31>"

SIG_START = "<6-0>"
SIG_END = "<6-253>"

TEMPO_START = "<7-0>"
TEMPO_END = "<7-48>"

SPECIAL_TOKENS = ['<mask>', '<s>', '<pad>', '</s>', '<unk>']


# We provide two modes to generate mask prediction
class PRED_MODE(Enum):
    VANILLA = 1
    OCTUPLE_MODE = 2


# Functions to return range of indices for an Octuple element
def bar_range(label_dict): return label_dict.index(
    BAR_START), label_dict.index(BAR_END)+1


def pos_range(label_dict): return label_dict.index(
    POS_START), label_dict.index(POS_END)+1


def ins_range(label_dict): return label_dict.index(
    INS_START), label_dict.index(INS_END)+1


def pitch_range(label_dict): return label_dict.index(
    PITCH_START), label_dict.index(PITCH_END)+1


def dur_range(label_dict): return label_dict.index(
    DUR_START), label_dict.index(DUR_END)+1


def vel_range(label_dict): return label_dict.index(
    VEL_START), label_dict.index(VEL_END)+1


def sig_range(label_dict): return label_dict.index(
    SIG_START), label_dict.index(SIG_END)+1


def tempo_range(label_dict): return label_dict.index(
    TEMPO_START), label_dict.index(TEMPO_END)+1

# The tokens should be in order (`0-bar`, `1-position`, `2-instrument`, `3-pitch`, `4-duration`, `5-velocity`, `6-timesig` , `7-tempo`) so we switch temperature value accordingly
# Limit to some specific fields such as pitch temp, duration temp, velocity temp, instrument temp


def switch_temperature(prev_index: int, label_dict, temperature_dict):
    """ Changes temperature to value for one of the eight fields in octuple 
        Args: 
          logits: logits distribution shape (vocabulary size)
          prev_index: previous predicted token 
          label_dict : dictionary mapping string octuple encodings to indices 
          temperature_dict : dict containing temperature values for all the 8 individual octuple elements 
        Returns: next temperature value 
    """
    # First we convert the token to it's string mapping
    prev_index = prev_index.item()
    rev_inv_map = reverse_label_dict(label_dict)
    str_encoding = rev_inv_map[prev_index]

    # print(((int(str_encoding[1]) + 1)%8))
    # print(str_encoding)

    return temperature_dict[((int(str_encoding[1]) + 1) % (8))]


def filter_invalid_indexes(logits, prev_index, label_dict, filter_value=-float('Inf')):
    """ Filter a distribution of logits using prev_predicted token 
          Args:
              logits: logits distribution shape (vocabulary size)
              prev_index: previous predicted token 
              label_dict : dictionary mapping string octuple encodings to indices 
        Returns: filtered logits according to prev_idx 
    """

    logits = logits.clone()

    prev_index = prev_index.item()
    rev_inv_map = reverse_label_dict(label_dict)
    str_encoding = rev_inv_map[prev_index]

    # For example if previous index was pitch than according to Octuple encoding next note should be duration
    # Therefore we fill up all the other 7 element ranges with infinity

    for tok in SPECIAL_TOKENS:
        logits[label_dict.index(tok)] = filter_value

    # if previous token was 'bar' then we mask everything excluding 'pos'
    if(str_encoding[1] == '0'):
        logits[list(range(*bar_range(label_dict)))] = filter_value
        logits[list(range(*ins_range(label_dict)))] = filter_value
        logits[list(range(*pitch_range(label_dict)))] = filter_value
        logits[list(range(*dur_range(label_dict)))] = filter_value
        logits[list(range(*vel_range(label_dict)))] = filter_value
        logits[list(range(*sig_range(label_dict)))] = filter_value
        logits[list(range(*tempo_range(label_dict)))] = filter_value
    # pos
    elif(str_encoding[1] == '1'):
        logits[list(range(*bar_range(label_dict)))] = filter_value
        logits[list(range(*pos_range(label_dict)))] = filter_value
        logits[list(range(*pitch_range(label_dict)))] = filter_value
        logits[list(range(*dur_range(label_dict)))] = filter_value
        logits[list(range(*vel_range(label_dict)))] = filter_value
        logits[list(range(*sig_range(label_dict)))] = filter_value
        logits[list(range(*tempo_range(label_dict)))] = filter_value
    # ins
    elif(str_encoding[1] == '2'):
        logits[list(range(*bar_range(label_dict)))] = filter_value
        logits[list(range(*pos_range(label_dict)))] = filter_value
        logits[list(range(*ins_range(label_dict)))] = filter_value
        logits[list(range(*dur_range(label_dict)))] = filter_value
        logits[list(range(*vel_range(label_dict)))] = filter_value
        logits[list(range(*sig_range(label_dict)))] = filter_value
        logits[list(range(*tempo_range(label_dict)))] = filter_value
    # pitch
    elif(str_encoding[1] == '3'):
        logits[list(range(*bar_range(label_dict)))] = filter_value
        logits[list(range(*pos_range(label_dict)))] = filter_value
        logits[list(range(*ins_range(label_dict)))] = filter_value
        logits[list(range(*pitch_range(label_dict)))] = filter_value
        logits[list(range(*vel_range(label_dict)))] = filter_value
        logits[list(range(*sig_range(label_dict)))] = filter_value
        logits[list(range(*tempo_range(label_dict)))] = filter_value
    # dur
    elif(str_encoding[1] == '4'):
        logits[list(range(*bar_range(label_dict)))] = filter_value
        logits[list(range(*pos_range(label_dict)))] = filter_value
        logits[list(range(*ins_range(label_dict)))] = filter_value
        logits[list(range(*pitch_range(label_dict)))] = filter_value
        logits[list(range(*dur_range(label_dict)))] = filter_value
        logits[list(range(*sig_range(label_dict)))] = filter_value
        logits[list(range(*tempo_range(label_dict)))] = filter_value
    # vel
    elif(str_encoding[1] == '5'):
        logits[list(range(*bar_range(label_dict)))] = filter_value
        logits[list(range(*pos_range(label_dict)))] = filter_value
        logits[list(range(*ins_range(label_dict)))] = filter_value
        logits[list(range(*pitch_range(label_dict)))] = filter_value
        logits[list(range(*dur_range(label_dict)))] = filter_value
        logits[list(range(*vel_range(label_dict)))] = filter_value
        logits[list(range(*tempo_range(label_dict)))] = filter_value
    # sig
    elif(str_encoding[1] == '6'):
        logits[list(range(*bar_range(label_dict)))] = filter_value
        logits[list(range(*pos_range(label_dict)))] = filter_value
        logits[list(range(*ins_range(label_dict)))] = filter_value
        logits[list(range(*pitch_range(label_dict)))] = filter_value
        logits[list(range(*dur_range(label_dict)))] = filter_value
        logits[list(range(*vel_range(label_dict)))] = filter_value
        logits[list(range(*sig_range(label_dict)))] = filter_value
    # tempo
    elif(str_encoding[1] == '7'):
        logits[list(range(*pos_range(label_dict)))] = filter_value
        logits[list(range(*ins_range(label_dict)))] = filter_value
        logits[list(range(*pitch_range(label_dict)))] = filter_value
        logits[list(range(*dur_range(label_dict)))] = filter_value
        logits[list(range(*vel_range(label_dict)))] = filter_value
        logits[list(range(*sig_range(label_dict)))] = filter_value
        logits[list(range(*tempo_range(label_dict)))] = filter_value

    return logits


def top_k_top_p(logits_batch, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """

    logits_batch = logits_batch.clone()

    # print(logits_batch.dim())

    if(logits_batch.dim() == 1):
        logits_batch = logits_batch.unsqueeze(0)

    # batch size 1 for now - could be updated for more but the code would be less clear
    assert logits_batch.dim() == 2

    # iterate through batch size
    for index, logits in enumerate(logits_batch):
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[
                0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[...,
                                     1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
    return logits_batch


def split_multi_oct(list_a, chunk_size):
    """
      Helper function for OCTUPLE_MODE and MULTI_OCTUPLE_MODE
      https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
    """

    for i in range(0, len(list_a), chunk_size):
        yield list(chain(*list_a[i:i + chunk_size]))


def predict_all_masks(roberta_model, roberta_label_dict, temperature_dict, octuplemidi_token_encoding: torch.Tensor, masked_octuples: list = None,
                      prediction_mode: PRED_MODE = PRED_MODE.VANILLA, mask_attribs: list = [3, 4, 5], num_multi_octuples: int = None,
                      temperature=1.0, top_k=30, top_p=0.6,
                      verbose=False):
    """
    Predict missing masks in sequence from left to right

    NOTE: Predicts ONLY the masked octuples provided in `masked_octuples` if `prediction_mode` is NOT Vanilla
    Else if prediction_mode is Vanilla, it predicts all the masks in the input `octuplemidi_token_encoding`

    octuplemidi_token_encoding: of the format torch.Tensor([0,0,0,0,0,0,0,0, ..........., 2,2,2,2,2,2,2,2]), where 0 is label_dict.bos_idx & 2 is label_dict.eos_idx
    prediction_mode: decides the speed of the mask prediction
    mask_attribs: decides which of the (`bar`, `position`, `instrument`, `pitch`, `duration`, `velocity`, `timesig` , `tempo`) are masked

    masked_octuples: List of the octuple indices in `octuplemidi_token_encoding` that are masked, note that the element indices for `bar` field of these elements would be (mask_octuple_idxs * 8)
    """

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
            torch.equal(
                octuplemidi_token_encoding[-8:], torch.Tensor([eos_idx]*8).type(tens_type))
    except:
        print('Start:', octuplemidi_token_encoding[:8])
        print(torch.Tensor([bos_idx]*8))
        print('End:', octuplemidi_token_encoding[-8:])
        print(torch.Tensor([bos_idx]*8))
        raise Exception(
            '`octuplemidi_token_encoding` either does not have 8 <s> tokens or 8 </s> tokens at beginning and end')

    # ---------------------------------------------------------------
    # Altering input mask list based on `prediction_mode`
    # ---------------------------------------------------------------

    mask_indices = None

    # If `masked_octuples` not provided, then the `prediction_mode` MUST be Vanilla
    if masked_octuples == None:
        try:
            assert prediction_mode == PRED_MODE.VANILLA
        except:
            # Since the current faster implementations involves the premise that in all the masked notes, same fields of each octuple is masked,
            # For example, we are not considering that in the sequence one octuple has just `duration` masked and another has just `pitch` masked
            raise Exception(
                "Error: Please choose `prediction_mode` as Vanilla since `masked_octuples` is not provided, to use faster modes provide `mask_indices`")

        mask_indices = [i for i, x in enumerate(
            octuplemidi_token_encoding.tolist()) if x == mask_idx]

    elif prediction_mode == PRED_MODE.VANILLA:

        print('Warning: Ignoring `masked_octuples`, `mask_attribs` & `num_multi_octuples` as `prediction_mode` is set as Vanilla')
        mask_indices = [i for i, x in enumerate(
            octuplemidi_token_encoding.tolist()) if x == mask_idx]

    elif prediction_mode == PRED_MODE.OCTUPLE_MODE:

        if num_multi_octuples is not None:
            print('Warning: Ignoring `num_multi_octuples` as `prediction_mode` is set as Octuple mode (not Multi-octuple mode)')

        mask_indices = [[x*8 + y for y in mask_attribs]
                        for x in masked_octuples]

    else:
        raise Exception("Invalid `prediction_mode`")

    try:
        assert len(mask_indices) > 0
    except AssertionError:
        raise Exception(
            'Please input sentence tokens with at least one mask token')

    try:
        assert all(torch.all(
            octuplemidi_token_encoding[octuple_midi_mask_elem] == mask_idx) for octuple_midi_mask_elem in mask_indices)
    except:
        print([octuplemidi_token_encoding[octuple_midi_mask_elem] ==
              mask_idx for octuple_midi_mask_elem in mask_indices])
        raise Exception(
            'Fatal error: At least one element of `mask_indices` is not <mask> (1236)')

    # --------------------------------------------------------------
    # Inputting masked indices to model using `prediction_strategy`
    # --------------------------------------------------------------

    if prediction_mode == PRED_MODE.VANILLA:
        pass

    elif prediction_mode == PRED_MODE.OCTUPLE_MODE:

        # Checking if `mask_attribs` is fine
        try:
            mask_attribs_len = len(mask_attribs)
            assert mask_attribs_len > 0 and \
                len(set(mask_attribs)) == mask_attribs_len and \
                all((x >= 0 and x < 8) for x in mask_attribs)
        except:
            raise Exception("`mask_attribs` not appropriate")

    print(f'Final mask indices list sent to model: {mask_indices}')
    print(
        f'Out of total input size: {len(octuplemidi_token_encoding.tolist())}')

    # Let's take an example where `mask_attribs` = [3,4,5] and at least

    # `final_mask_indices` is of form [3,4,5,11,12,13....] in `Vanilla` prediction mode
    # `final_mask_indices` is of form [[3,4,5],[11,12,13]......] in `Octuple` prediction mode
    # `final_mask_indices` is of form [[3,4,5,11,12,13]......] in `Multi Octuple` prediction mode, in this case num_multi_octuples = 2

    filter_value = -float('Inf')
    octuplemidi_token_encoding_device = octuplemidi_token_encoding.device

    # Finally predicting masks based on `prediction_strategy`

    ################################
    # Vanilla Mode Prediction mode #
    ################################

    if prediction_mode == PRED_MODE.VANILLA:

        repeat_count = 0

        for mask_idx_batch in tqdm(mask_indices):
            input = octuplemidi_token_encoding.unsqueeze(0).cuda()
            prev_idx = octuplemidi_token_encoding[mask_idx_batch-1]

            with torch.no_grad():
                # extr_features shape -> [1, 8016, 1237]
                # Here 1 is the batch size, 8016 is embedding dimension and 1237 is the vocab size
                extr_features, _ = roberta_model.model.extract_features(input)
                # filter the mask indices from the extracted feature tokens (1000 octuples)
                logits = extr_features[0, mask_idx_batch]

                temperature = switch_temperature(
                    prev_idx, roberta_label_dict, temperature_dict)
                repeat_penalty = max(
                    0, np.log((repeat_count+1)/4)/5) * temperature
                temperature += repeat_penalty

                if temperature != 1.:
                    logits = logits/temperature

                logits = filter_invalid_indexes(
                    logits, prev_idx, label_dict, filter_value=filter_value)
                logits = top_k_top_p(logits, top_k=top_k,
                                     top_p=top_p, filter_value=filter_value)

                probs = torch.softmax(logits, dim=-1)
                if probs.dim() == 1:
                    probs = probs.unsqueeze(0)

                # Update repeat count
                num_choices = len(probs.nonzero().reshape(-1))
                if num_choices <= 2:
                    repeat_count += 1
                else:
                    repeat_count = repeat_count // 2

                if(temperature != 1 or top_k != 0 or top_p != 0):
                    # We sample from a multinomial distribution
                    top_preds = torch.multinomial(probs, 1)
                    top_preds = top_preds.reshape(-1)
                else:
                    # We take the argmax or only choose the top candidate
                    # print('Predicting argmax mode!')
                    top_preds = torch.argmax(probs, dim=1)

            # Assign the token
            octuplemidi_token_encoding_type = octuplemidi_token_encoding.dtype
            octuplemidi_token_encoding[mask_idx_batch] = top_preds.\
                type(octuplemidi_token_encoding_type). \
                to(octuplemidi_token_encoding_device)

    ################################
    # Octuple Mode Prediction mode #
    ################################

    # For Octuple mod switching temperature is not possible so we only use one value
    elif prediction_mode == PRED_MODE.OCTUPLE_MODE:

        # iterate through all the mask indices
        for mask_idx_batch in tqdm(mask_indices):
            input = octuplemidi_token_encoding.unsqueeze(0).cuda()

            with torch.no_grad():
                # extr_features shape -> [1, 8016, 1237]
                # Here 1 is the batch size, 8016 is embedding dimension and 1237 is the vocab size
                extr_features, _ = roberta_model.model.extract_features(input)
                # filter the mask indices from the extracted feature tokens (1000 octuples)
                logits = extr_features[0, mask_idx_batch]

                # Apply Temperature if not equal to 1
                if temperature != 1.:
                    logits = logits/temperature

                # Apply top-k and top-p if != 0
                logits = top_k_top_p(logits, top_k=top_k,
                                     top_p=top_p, filter_value=filter_value)

                probs = torch.softmax(logits, dim=-1)
                if probs.dim() == 1:
                    probs = probs.unsqueeze(0)

                if(temperature != 1 or top_k != 0 or top_p != 0):
                    # We sample from a multinomial distribution
                    top_preds = torch.multinomial(probs, 1)
                    top_preds = top_preds.reshape(-1)
                else:
                    # We take the argmax or only choose the top candidate
                    top_preds = torch.argmax(probs, dim=1)

            octuplemidi_token_encoding_type = octuplemidi_token_encoding.dtype
            octuplemidi_token_encoding[mask_idx_batch] = top_preds.\
                type(octuplemidi_token_encoding_type). \
                to(octuplemidi_token_encoding_device)

    # Final error check
    try:
        assert not any(torch.all(
            octuplemidi_token_encoding[octuple_midi_mask_elem] == mask_idx) for octuple_midi_mask_elem in mask_indices)
    except:
        print([octuplemidi_token_encoding[octuple_midi_mask_elem] ==
              mask_idx for octuple_midi_mask_elem in mask_indices])
        raise Exception(
            'Fatal error: The prediction has at least one element of `mask_indices` as <mask>')

    return octuplemidi_token_encoding
