import miditoolkit

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
def filter_tracks(midi_obj: miditoolkit.midi.parser.MidiFile, root_folder:str = '.', 
                    cache_folder:str = "musicbert_cache", midi_name:str = "final", track_ids:list = []):
  
  if len(track_ids) > 0:
    new_midi_obj = miditoolkit.midi.parser.MidiFile()
    new_midi_obj.instruments = [midi_obj.instruments[i] for i in track_ids]

    new_midi_obj.ticks_per_beat = midi_obj.ticks_per_beat
    new_midi_obj.time_signature_changes = midi_obj.time_signature_changes
    new_midi_obj.tempo_changes = midi_obj.tempo_changes

    midi_obj = new_midi_obj
    midi_obj.dump(f'{root_folder}/{cache_folder}/{midi_name}_track_filtered.mid')

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
