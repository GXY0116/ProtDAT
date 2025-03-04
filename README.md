# ProtDAT
A Unified Framework for Protein Sequence Design from Any Protein Text Description


## Preparation
ProtDAT is implemented with Python3 (>=3.9). We recommend you to use a virtual environment to install the dependencies :
``` bash
conda create -n ProtDAT python=3.9
```
PyTorch can be installed by selecting the corresponding version through ```https://pytorch.org/```. 
After that, install other requirements by :
```bash
pip install -r requirements.txt
```
Finally, activate the virtual environment by :
```bash
conda activate ProtDAT
```

## Download models and files
Before using ProtDAT, there are several steps:

1. Download the [ESM1b](https://huggingface.co/facebook/esm1b_t33_650M_UR50S) and [PubMedBERT](https://huggingface.co/NeuML/pubmedbert-base-embeddings) models and place them in the ```esm1b``` and ```pubmedbert``` subfolders within the ```model``` directory.
2. Download ProtDAT model weight file [state_dict.pth](https://zenodo.org/records/14264096) and [datasets](https://zenodo.org/records/14967237).


## Usage
### Generate protein sequences with protein descriptions (and protein sequence fragments)
For generating protein sequences one by one or in batches, separately refer to ```gen_single_seq.py``` and ```gen_batch_seqs.py```.

The cases of protein sequences and text descriptions are in the ```data``` directory. For example :
```bash
Description: FUNCTION: Component of the acetyl coenzyme A carboxylase complex. SUBCELLULAR LOCATION: Cytoplasm. SIMILARITY: Belongs to the AccA family.
Sequence: MAVSDRKLQLLDFEKPLAELEDRIEQIRSLSEQNGVDVTDQIAQLEGRAEQLRQEIFSSLTPMQELQLARHPRRPSTLDYIHAISDEWMELHGDRRGYDDPAIVGGVGRIGGQPVLMLGHQKGRDTKDNVARNFGMPFPSGYRKAMRL...
```
The generation codes below determine whether the process is guided solely by text or by a combination of text and sequence.
```python
seq=None,                                           # Only protein descriptions guide the generation process
seq=tokenized_seqs['input_ids'][...,:1].to(device), # Both sequence fragments and descriptions guide the generation process
```
### Train new model based on ProtDAT
You can build a custom protein text-sequence dataset with a specific pattern and train it using the architecture in ```Decoder.py```.


## Citations
If you find ProtDAT useful, cite the relevant paper:
```bibtex
@article{guo2024protdat,
  title={ProtDAT: A Unified Framework for Protein Sequence Design from Any Protein Text Description},
  author={Guo, Xiao-Yu and Li, Yi-Fan and Liu, Yuan and Pan, Xiaoyong and Shen, Hong-Bin},
  journal={arXiv preprint arXiv:2412.04069},
  year={2024}
}
```


## License <a name="license"></a>
### Code License
1. The ProtDAT source codes are licensed under the [MIT license](https://github.com/GXY0116/ProtDAT/blob/main/LICENSE).
2. The ESM1b model can be found at [ESM1b](https://github.com/facebookresearch/esm), which is under the [MIT license](https://github.com/facebookresearch/esm/blob/main/LICENSE)
3. The PubMedBERT model can be found at [PubMedBERT](https://huggingface.co/NeuML/pubmedbert-base-embeddings/tree/main), which is under the [Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)

### Model Parameters License
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
The [ProtDAT parameters](https://zenodo.org/records/14264096) are made availabe under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
