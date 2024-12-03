# ProtDAT
A Unified Framework for Protein Sequence Design from Any Protein Text Description

## Preparation
ProtDAT is implemented with Python3 (>=3.9), codes can be downloaded by :
```bash
git clone https://github.com/GXY0116/ProtDAT.git
```
We recommend you to use a virtual environment, such as Anaconda, to install the dependencies :
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

## Usage
### Generate protein sequences with protein descriptions (and protein sequence fragments)
For generating protein sequences one by one or in batches, separately refer to ```gen_single_seq.py``` and ```gen_batch_seqs.py```.

The cases of protein sequences and text descriptions are in the ```data``` directory. For example :
```bash
Description: FUNCTION: Component of the acetyl coenzyme A carboxylase complex. SUBCELLULAR LOCATION: Cytoplasm. SIMILARITY: Belongs to the AccA family.
Sequence: MAVSDRKLQLLDFEKPLAELEDRIEQIRSLSEQNGVDVTDQIAQLEGRAEQLRQEIFSSLTPMQELQLARHPRRPSTLDYIHAISDEWMELHGDRRGYDDPAIVGGVGRIGGQPVLMLGHQKGRDTKDNVARNFGMPFPSGYRKAMRL...
```

## License <a name="license"></a>
### Code License
The ProtDAT source codes are licensed under the MIT license.

The ESM1b model can be found at [ESM1b](https://github.com/facebookresearch/esm), which is under the [MIT license](https://github.com/facebookresearch/esm/blob/main/LICENSE)

The PubMedBERT model can be found at [PubMedBERT](https://huggingface.co/NeuML/pubmedbert-base-embeddings)
### Model Parameters License
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
The ProtDAT parameters are made availabe under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
