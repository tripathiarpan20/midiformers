from preprocess import *
from utils import *
from musicbert_model import *

import os

if __name__ == '__main__':
  print('Testing pipeline.py')

  os.system("rm -r /musicbert_cache/")
  os.system("mkdir -p /musicbert_cache/")

  # TODO : Complete the CLI script 

  # midi_obj, midi_name = parse_midi_file(sample_midi_path)
  # cache_midi_tracks(midi_obj, midi_name, '/content', 'musicbert_cache')

'''
Sample pipepline:

midi_obj, midi_name = parse_midi_file(sample_midi_path)
cache_midi_tracks(midi_obj, midi_name, '/content', 'musicbert_cache')

filter_track_ids = []
midi_obj =  filter_tracks(midi_obj, '.', 'musicbert_cache', midi_name, track_ids)
e = MIDI_to_encoding(midi_obj) #MIDI_to_encoding is from preprocess.py
label_dict = MusicBERTModel.task.label_dictionary
octuple_midi_str = encoding_to_stsr(e)
encoding = label_dict.encode_line(octuple_midi_str)

#Choosing mask parameters

min_bar = int(input('Enter minimum bar length to be left unmasked: '))

avail_programs = set([oct[2] for oct in e[int(min_idx/8):1000]])
min_idx = get_min_bar_idx_from_oct(octuple_midi_str_aslist, min_bar_mask)
avail_programs = set([oct[2] for oct in e[int(min_idx/8):1000]])
print(f"Programs(instruments) available in MIDI due to internal clipping to 1000 octuples::{avail_programs}")

program_id = int(input('Enter program ID to mask from last output: ')) #Choose which program(instrument) to mask
program_percentage_mask = float(input('Enter percentage of program to mask: ')) #Choose which program(instrument) to mask

masked_encoding = encoding.clone()

repl_flag = input('Do you want to replace the masked program with another program? (y/n): ')
if repl_flag == 'y':
    #Change to another program ID if you want to mask other instruments
    replacement_program_id = int(input('Enter program ID to replace the masked program: '))
elif repl_flag == 'n':
    replacement_program_id = program_id
else:
    raise ValueError('Invalid input')

#Change to mask other attributes of octuple (0 Measure, 1 Pos, 2 Program, 3 Pitch, 4 Duration, 5 Velocity, 6 TimeSig, 7 Tempo)
mask_attribs = [3,4,5,6,7]

#Masking percentage of song
masked_encoding, masked_oct_idxs = mask_instrument_notes_program(program_id, masked_encoding, label_dict, program_percentage_mask,
                                            replacement_program = replacement_program_id, mask_attribs = mask_attribs,
                                            min_bar_mask = min_bar_mask)

#Choose prediction strategy
prediction_strategy = input("Choose prediction strategy out of \"Vanilla\", \"Octuple\", \"Multi-octuple\": ) 
#Choose prediction strategy out of "Vanilla", "Octuple", "Multi-octuple"

num_multi_octuples = None
if prediction_strategy == 'Multi-octuple':
    num_multi_octuples = int(input('Enter number of multi-octuples to predict if chosen \"Multi-octuple\" above: '))

prediction_mode = None
if prediction_strategy == "Vanilla":
  prediction_mode = PRED_MODE.VANILLA
elif prediction_strategy == "Octuple":
  prediction_mode = PRED_MODE.OCTUPLE_MODE
elif prediction_strategy == "Multi-octuple":
  #@markdown If prediction_strategy is `Multi-octuple`, choose the number of octuples.
  #@markdown `num_multi_octuples` will be ignored if `prediction_strategy` is different
  num_multi_octuples =  3#@param {type:"integer"}
  prediction_mode = PRED_MODE.MULTI_OCTUPLE_MODE

#Predict masks
mask_prediction_output = predict_all_masks(roberta_base, label_dict, masked_encoding,
                          masked_octuples = masked_oct_idxs, mask_attribs = mask_attribs,
                          prediction_mode = prediction_mode,  num_multi_octuples = num_multi_octuples)


mask_prediction_output_str = decode_w_label_dict(label_dict , mask_prediction_output )

out_midi_enc = str_to_encoding(mask_prediction_output_str)
out_midi_obj = encoding_to_MIDI( out_midi_enc )

save_output_midi(out_midi_obj, prediction_strategy, 
    program_id, replacement_program_id, 
    program_percentage_mask, midi_name,
    '/content', cache_folder = 'musicbert_cache')

'''
