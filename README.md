### Getting Started

```bash
python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
pip install gym[box2d]

cd part1
python train.py testing=true save_video=false
```
### For Contributors

Using `script.sh` to push to git:

```bash
source ./script.sh <commit-message-here>
```

