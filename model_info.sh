source .venv/bin/activate
python3 -c "
import sys, os, torch
sys.path.insert(0, 'src')
m = torch.load('models/v0.pt', weights_only=False, map_location='cpu')
nb = len(m.blocks)
nf = m.primary_pegs.weight.shape[0]
total = sum(p.numel() for p in m.parameters())
print(f'num_blocks  = {nb}')
print(f'num_filters = {nf}')
print(f'total params= {total:,}')
print(f'file size   = {os.path.getsize(\"models/v0.pt\")/1e6:.1f} MB')
"
