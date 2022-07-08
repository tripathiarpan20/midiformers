from preprocess import *
from utils import *
from musicbert_model import *


def parse_midi_file(sample_midi_path: str):
  midi_obj = miditoolkit.midi.parser.MidiFile(sample_midi_path)
  midi_name = sample_midi_path.split('/')[-1].split('.')[0]
  return midi_obj, midi_name


def cache_midi_tracks(midi_obj: miditoolkit.midi.parser.MidiFile, midi_name, root_folder = '.', 
                      cache_folder = "musicbert_cache", verbose = False):
  instrument_progs = []
  
  for track_idx, ins_track in enumerate(midi_obj.instruments):  
    temp_midi_obj = miditoolkit.midi.parser.MidiFile()
    temp_midi_obj.instruments = [ins_track]

    temp_midi_obj.ticks_per_beat = midi_obj.ticks_per_beat
    temp_midi_obj.time_signature_changes = midi_obj.time_signature_changes
    temp_midi_obj.tempo_changes = midi_obj.tempo_changes

    instrument_progs.append(ins_track.program)
    temp_midi_obj.dump(f'{root_folder}/{cache_folder}/{midi_name}_track_{track_idx}_prog_{ins_track.program}.mid')
  
  if verbose:
    print(f'The input MIDI has {len(midi_obj.instruments)} tracks with program IDs {instrument_progs} respectively.')



'''
set `track_ids` as a list of indices of track IDs to keep, else leave it as [] for taking all tracks in MIDI
'''
def filter_tracks(midi_obj: miditoolkit.midi.parser.MidiFile, root_folder = '.', 
                    cache_folder = "musicbert_cache", midi_name:str = "final", track_ids:list = []):
  
  if len(track_ids) > 0:
    new_midi_obj = miditoolkit.midi.parser.MidiFile()
    new_midi_obj.instruments = [midi_obj.instruments[i] for i in track_ids]

    new_midi_obj.ticks_per_beat = midi_obj.ticks_per_beat
    new_midi_obj.time_signature_changes = midi_obj.time_signature_changes
    new_midi_obj.tempo_changes = midi_obj.tempo_changes

    midi_obj = new_midi_obj
    midi_obj.dump(f'/content/{midi_name}_track_filtered.mid')

    print(f"> Parsed MIDI: {midi_name}")
    print(f"> Saved Parsed MIDI as : {root_folder}/{cache_folder}/{midi_name}_track_filtered.mid")
    # playMidi(f'/content/{midi_name}_track_filtered.mid')
    
  return midi_obj

def save_output_midi(out_midi_obj, prediction_strategy,
    program_id, replacement_program_id,
    program_percentage_mask, midi_name, 
    root_folder = '.', cache_folder = 'musicbert_cache'):

    out_path = None

    if program_id == replacement_program_id:
        if program_percentage_mask == 100: 
            out_path = f'{root_folder}/{cache_folder}/{midi_name}_fullprogram{program_id}_maskpred_{prediction_strategy}.mid'
            out_midi_obj.dump(out_path)
        else:
            out_path = f'{root_folder}/{cache_folder}/{midi_name}_program{program_id}_{program_percentage_mask}percentmaskpred_minbarmask{min_bar_mask}_{prediction_strategy}.mid'
            out_midi_obj.dump(out_path)
    else:
        out_path = f'{root_folder}/{cache_folder}/{midi_name}_fullprogram{program_id}masked_program{replacement_program_id}pred_{prediction_strategy}.mid'
        out_midi_obj.dump(out_path)
    print(f"> Output MIDI saved to path {out_path}")

if __name__ == '__main__':
  print('Testing pipeline.py')


'''
Sample pipepline:

midi_obj, midi_name = parse_midi_file(sample_midi_path)
cache_midi_tracks(midi_obj, midi_name, '/content', 'musicbert_cache')

filter_track_ids = []
midi_obj =  filter_tracks(midi_obj, '.', 'musicbert_cache', midi_name, track_ids)
e = MIDI_to_encoding(midi_obj) #MIDI_to_encoding is from preprocess.py
label_dict = MusicBERTModel.task.label_dictionary
octuple_midi_str = encoding_to_str(e)
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
