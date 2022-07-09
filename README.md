# MIDIformers

## Table of Contents
- [MIDIformers](#midiformers)
  - [Table of Contents](#table-of-contents)
  - [DeepMixer <a name="deepmixer"></a>](#deepmixer-)
    - [Installation <a name="deepmixerinstall"></a>](#installation-)
    - [Notebook <a name="deepmixernotebook"></a>](#notebook-)
    - [Output samples <a name="deepmixeroutsamples"></a>](#output-samples-)

## DeepMixer <a name="deepmixer"></a> 

The project involves the usage of the open-source [MusicBERT](https://github.com/microsoft/muzic/tree/main/musicbert) model to perform mask prediction tasks for MIDI task with customizability.

### Installation <a name="deepmixerinstall"></a> 

```
git clone https://github.com/tripathiarpan20/midiformers.git
cd midiformers/DeepMixer/models/musicbert
./setup.sh
```

### Notebook <a name="deepmixernotebook"></a> 
The live version of Colab notebook utilising the scripts in `DeepMixer/models/musicbert` can be accessed from this link : 

<a href="https://colab.research.google.com/drive/1C7jS-s1BCWLXiCQQyvIl6xmCMrqgc9fg?usp=sharing">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Image" style="display: block; margin: 0 auto" />
</a>

The notebook supports customisability on top of the original MusicBERT codebase, like masking chosen percentage of random notes from either whole MIDI stream/ notes from selected instruments based on user preference. 

Other features include: 

- [x] Option to leaving notes from the beginning `min_bar_mask` masks out of the masking pool to provide more initial context for mask prediction.  
- [x] Prediction modes with trade-off between speed and quality of predictions.

To be implemented: 
- [ ] Prediction strategies like Top-k and Temperature for mask predictions. 
- [ ] Track-level masking. 


### Output samples <a name="deepmixeroutsamples"></a> 
Some of the samples from the above notebook along with the reference pieces can be found in a [Drive folder](https://drive.google.com/drive/folders/1fO9zKbwMHwDgy_p84q5dM5yFf0y8f9y9?usp=sharing).

A few of our favorites are embedded below:  

* Shock (Attack on Titan):
    - Original & Remix: 

https://user-images.githubusercontent.com/42506819/178094976-a5763a31-70bf-4a17-b9e9-1255c0b9a1f0.mp4

    


https://user-images.githubusercontent.com/42506819/178094989-441471d2-78ce-4b6a-8e88-bd01aeab810e.mp4


* Bohemian Rhapsody (Queen):
    - Original & Remix: 
    
 

https://user-images.githubusercontent.com/42506819/178095093-1e1192bc-76e5-4027-b733-d63bed9ac206.mp4



https://user-images.githubusercontent.com/42506819/178095109-b9a38d56-3b69-4169-9ed1-3eb82a47276a.mp4


* Unforgiven 2 (Metallica): 
    - Original & Remix: 
    


https://user-images.githubusercontent.com/42506819/178095316-6d19c803-3407-4bbd-81e1-0dc476b680d6.mp4



https://user-images.githubusercontent.com/42506819/178095326-f6961d98-c574-4ee2-8d27-27efead79409.mp4


    
