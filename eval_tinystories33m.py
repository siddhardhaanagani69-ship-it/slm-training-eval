import math, json, numpy as np, torch, torch.nn.functional as F, tiktoken
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

BLOCK, BS = 128, 64
dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
enc = tiktoken.get_encoding('gpt2')
ds = load_dataset('roneneldan/TinyStories', split='validation')
ids = enc.encode_ordinary_batch(ds['text'])
val = np.concatenate([np.array(x, dtype=np.uint16) for x in ids])
assert len(val) == 4_743_928, len(val)  # same stream as the notebook's val.bin

tok = AutoTokenizer.from_pretrained('roneneldan/TinyStories-33M')
for t in ds['text'][:200]:
    assert tok(t)['input_ids'] == enc.encode_ordinary(t)
print('tokenizer identical to tiktoken gpt2 on 200 stories')

m = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M', dtype=torch.float32).to(dev).eval()
tot_p = sum(p.numel() for p in m.parameters())
emb = sum(p.numel() for n, p in m.named_parameters() if 'wte' in n or 'wpe' in n)
print(f'params total {tot_p/1e6:.1f}M, non-embedding {(tot_p-emb)/1e6:.1f}M')
print(m.config)

n = (len(val) - 1) // BLOCK
tot, cnt = 0.0, 0
with torch.no_grad():
    for s in range(0, n, BS):
        idx = np.arange(s, min(s + BS, n)) * BLOCK
        x = torch.from_numpy(np.stack([val[i:i+BLOCK] for i in idx]).astype(np.int64)).to(dev)
        y = torch.from_numpy(np.stack([val[i+1:i+1+BLOCK] for i in idx]).astype(np.int64)).to(dev)
        logits = m(x).logits.float()
        tot += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum').item(); cnt += y.numel()
loss = tot / cnt
print(f'TinyStories-33M full-val loss {loss:.3f} ppl {math.exp(loss):.2f} over {cnt:,} tokens')

out = {'val_loss': loss, 'val_ppl': math.exp(loss), 'params': tot_p, 'non_emb': tot_p - emb, 'samples': []}
for p in ['Once upon a time there was a pumpkin.', 'A little girl went to the woods', 'Tom was sad because']:
    torch.manual_seed(0)
    g = m.generate(torch.tensor([enc.encode_ordinary(p)], device=dev), max_new_tokens=120, do_sample=True,
                   temperature=0.8, top_k=50, pad_token_id=50256)
    s = enc.decode(g[0].tolist()); out['samples'].append(s); print('---', p, '\n', s)
json.dump(out, open('tinystories33m_results.json', 'w'), indent=1)
