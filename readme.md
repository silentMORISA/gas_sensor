# Installation
Started with an environment of python==3.8, install pytorch first:

```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

Then enter the local repository of `segment-anything` and install it:

```
cd segment-anything; pip install -e .
```

Finally, install other necessary packages:

```
pip install -r requirement.txt
```

# Getting started

You can use following command to run this project:

```
python sam_official.py --image_dir /path/to/your/images --save_dir /path/to/save/outputs --model_type vit_l --weights /data/basemodel/sam_vit_l_0b3195.pth
```

Basically you only need to modify `image_dir` and `save_dir` to your own path.

Your directory containing input images should hold this structure:

```
image_dir/
├── patient/
│   ├── before/
│   └── after/
├── normal/
│   ├── before/
│   └── after/
```

<hr>
<div align="center">
  <b>JUN XIAO</b><br>
  <i>PhD student of HIT/GBU</i><br>
  📧 jxiaoas@connect.ust.hk
</div>