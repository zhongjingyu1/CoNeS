## VG256

1. Navigate to the VG256 data directory:
```
cd ./vg256
```
2. Download the data:
```
curl https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip --output images.zip
curl https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip --output images2.zip
curl https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects_v1_2.json.zip --output objects.zip
```
3. Extract and merge the data:
```
unzip -q images.zip
unzip -q images2.zip
unzip -q objects.zip
mv ./VG_100K_2/* ./VG_100K/
```
4. Format the data (If the `formatted_xxx_xxx.npy` files already exist, this step can be skipped.):
```
python format_vg256.py
```
5. Clean up:
```
rm images.zip
rm images2.zip
rm objects.zip
rm -rf VG_100K_2
```

## Data directory for VG256.

After following all of the data preparation instructions, the top level of this directory should consist of the following files and directories:
* `VG_100K/`
* `format_vg256.py`
* `formatted_train_images.npy`
* `formatted_train_labels.npy`
* `formatted_val_images.npy`
* `formatted_val_labels.npy`
* `objects.json`
* `README.md`
* `vg256.json`

## Acknowledgements
We replicated the Visual Genome dataset version (VG-256) from [MLC-PAT](https://github.com/xiemk/MLC-PAT). We thank the authors for releasing their code.
