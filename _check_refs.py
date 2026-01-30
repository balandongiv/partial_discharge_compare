import re, glob, os
tex_files = glob.glob('latex/**/*.tex', recursive=True)
ref_cmds = ['ref','pageref','autoref','eqref','cref','Cref','nameref']
ref_re = re.compile(r"\\(?:" + "|".join(ref_cmds) + r")\{([^}]+)\}")
label_re = re.compile(r"\\label\{([^}]+)\}")
refs = {}
labels = {}
for f in tex_files:
    s = open(f, 'r', encoding='utf-8', errors='ignore').read()
    for m in ref_re.finditer(s):
        refs.setdefault(m.group(1), []).append(f)
    for m in label_re.finditer(s):
        labels.setdefault(m.group(1), []).append(f)
missing = sorted([k for k in refs if k not in labels])
print('Unique refs(all ref-like):', len(refs))
print('Missing labels for refs:', len(missing))
for k in missing[:120]:
    files = sorted(set(os.path.relpath(p) for p in refs[k]))
    print(f'- {k}  (in {"; ".join(files)})')
if len(missing) > 120:
    print('... truncated ...')
