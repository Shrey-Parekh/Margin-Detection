"""Score the detector against the 29 expert-labelled papers.

Assumes rows 1-29 of the image set correspond to features_manual_29.csv in
order, as confirmed with the project owner.
"""
import pandas as pd

LEFT = ['SLM', 'WLM', 'DAFLM', 'RFLM', 'CCLM', 'CVLM']
TOP = ['TT', 'TS', 'TDA']
BOT = ['BT', 'BS', 'BDA']

auto = pd.read_csv('features_auto_full.csv')
man = pd.read_csv('features_manual_29.csv')
man.columns = [c.upper() for c in man.columns]
man = man.rename(columns={'RLM': 'RFLM'})

a = auto.iloc[:29].reset_index(drop=True)

print(f"detector rows: {len(auto)}   compared against expert rows: {len(man)}\n")
print("=== coverage ===")
print(f"  left_valid   : {auto['left_valid'].sum()}/{len(auto)}")
print(f"  top_valid    : {auto['top_valid'].sum()}/{len(auto)}")
print(f"  bottom_valid : {auto['bottom_valid'].sum()}/{len(auto)}")

print("\n=== class distribution (all detector rows vs expert 29) ===")
for name, cols in [('LEFT', LEFT), ('TOP', TOP), ('BOT', BOT)]:
    print(f"-- {name}")
    for c in cols:
        print(f"   {c:6s} detector {auto[c].mean() * 100:5.1f}%   expert {man[c].mean() * 100:5.1f}%")

print("\n=== category agreement on the 29 ===")
for name, cols in [('left', LEFT), ('top', TOP), ('bot', BOT)]:
    valid = a['left_valid'] == 1 if name == 'left' else (
        a['top_valid'] == 1 if name == 'top' else a['bottom_valid'] == 1)
    e = man[cols].values.argmax(1)
    d = a[cols].values.argmax(1)
    m = valid.values & (a[cols].sum(axis=1).values > 0)
    chance = 100.0 / len(cols)
    if m.sum():
        print(f"  {name:5s}: {(e[m] == d[m]).mean() * 100:5.1f}%  on n={m.sum():2d} "
              f"(chance {chance:.0f}%)")
    else:
        print(f"  {name:5s}: no comparable rows")

print("\n=== confusion: rows=expert, cols=detector ===")
for name, cols in [('LEFT', LEFT), ('TOP', TOP), ('BOT', BOT)]:
    e = man[cols].values.argmax(1)
    d = a[cols].values.argmax(1)
    M = pd.DataFrame(0, index=[f'exp_{c}' for c in cols],
                     columns=[f'det_{c}' for c in cols])
    for i, j in zip(e, d):
        M.iloc[i, j] += 1
    print(f"-- {name}")
    print(M.to_string())

print("\n=== deadband sweep: which threshold best reproduces expert SLM calls? ===")
sub = a[a['left_valid'] == 1].copy()
exp_slm = man.loc[sub.index, 'SLM'].values
best = None
for tau in range(0, 61, 2):
    pred = ((sub['left_spread_px'] <= tau) & (sub['left_scatter_px'] <= tau)).astype(int).values
    acc = (pred == exp_slm).mean()
    if best is None or acc > best[1]:
        best = (tau, acc)
    if tau % 6 == 0:
        print(f"   tau={tau:3d}px  agreement with expert SLM = {acc * 100:5.1f}%  "
              f"(detector says straight {pred.mean() * 100:.0f}% of the time)")
print(f"\n   best tau = {best[0]}px at {best[1] * 100:.1f}% agreement "
      f"(current setting is {__import__('extract_features').SLM_DEADBAND_PX}px)")
