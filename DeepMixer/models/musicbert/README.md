# MusicBERT
[MusicBERT: Symbolic Music Understanding with Large-Scale Pre-Training](https://arxiv.org/pdf/2106.05630.pdf), by Mingliang Zeng, Xu Tan, Rui Wang, Zeqian Ju, Tao Qin, Tie-Yan Liu, ACL 2021, is a large-scale pre-trained model for symbolic music understanding. It has several mechanisms including OctupleMIDI encoding and bar-level masking strategy that are specifically designed for symbolic music data, and achieves state-of-the-art accuracy on several music understanding tasks, including melody completion, accompaniment suggestion, genre classification, and style classification. 

# Commit information 

**Working with the folder `muzic/musicbert` of the commit https://github.com/microsoft/muzic/tree/f833b4f383e3df06235913fd50d94c2600e7af23 for this project.** 

Added scripts: 
- `utils/utils_label.py`
- `utils/utils_masking.py`
- `utils/utils_mask_prediction.py`
- `pipeline.py`