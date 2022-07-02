# DeepMusicGeneration 

Generating long pieces of music is a challenging problem, as music contains structure at multiple timescales, from milisecond timings to motifs to phrases to repetition of entire sections. We used Transformer XL, an attention-based neural network that can generate music with improved long-term coherence.

While the original Transformer allows us to capture self-reference through attention, it relies on absolute timing signals and thus has a hard time keeping track of regularity that is based on relative distances, event orderings, and periodicity. Whereas the Transformer XL model uses relative attention, which explicitly modulates attention based on how far apart two tokens are, the model is able to focus more on relational features. Relative self-attention also allows the model to generalize beyond the length of the training examples, which is not possible with the original Transformer model.

The model is powerful enough, that it learns abstractions of data on its own, without much human-imposed domain knowledge or constraints. In contrast with this general approach,
our implementation shows that Transformers can do even better for music modeling, when we improve the way a musical score is converted into the data fed to a Transformer model. 


