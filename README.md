# MIDIformers

[**Sample outputs**](https://drive.google.com/drive/folders/18_Ze8gJir6F3feQV6egZzNdmrrLYTYh8?usp=sharing) | [**Video demo**](https://player.vimeo.com/video/709351568?h=8ea1cae660&amp;badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479) | [**PDF report**](https://drive.google.com/file/d/11x6lg66kgqX9fL7dy3ZyMIyFBjW8aK-0/view?usp=sharing)

## Table of Contents
* [Introduction](#intro)
* [Acknowledgements](#ackn)
* [Methodology](#methods)

    - [Preprocessing](#preproc)
    - [Models](#models)
    - [Tasks](#tasks)
* [Dataset](#dataset)
* [Training](#training)
* [Evaluation](#eval) 

    - [Pretrained models](#training)
    - [Running Streamlit app](#streamlitapp)

## Introduction <a name="intro"></a> 

Generating long pieces of music is a challenging problem, as music contains structure at multiple timescales, from milisecond timings to motifs to phrases to repetition of entire sections. We used Transformer XL, an attention-based neural network that can generate music with improved long-term coherence.

While the original Transformer allows us to capture self-reference through attention, it relies on absolute timing signals and thus has a hard time keeping track of regularity that is based on relative distances, event orderings, and periodicity. Whereas the Transformer XL model uses relative attention, which explicitly modulates attention based on how far apart two tokens are, the model is able to focus more on relational features. Relative self-attention also allows the model to generalize beyond the length of the training examples, which is not possible with the original Transformer model.

The model is powerful enough, that it learns abstractions of data on its own, without much human-imposed domain knowledge or constraints. In contrast with this general approach,
our implementation shows that Transformers can do even better for music modeling, when we improve the way a musical score is converted into the data fed to a Transformer model. 

## Acknowledgements <a name="ackn"></a> 

This project was made possible with previous contributions referenced below: 
<ol>
  <li> https://github.com/bearpelican/musicautobot/ </li>
  <li> https://web.mit.edu/music21/ </li>
  <li> https://streamlit.io/ </li>
</ol>

## Methodology <a name="methods"></a> 

### Preprocessing <a name="preproc"></a>

### Models <a name="models"></a>

### Tasks <a name="tasks"></a>

## Dataset <a name="dataset"></a> 

## Training <a name="training"></a> 

## Evaluation <a name="eval"></a> 

### Pretrained models <a name="modellinks"></a> 

## Running Streamlit app <a name="streamlitapp"></a> 
