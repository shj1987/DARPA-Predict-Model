import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--i_path", "-i", type=str, default='./utils.py')
parser.add_argument("--o_path", "-o", type=str, required=True)
args = parser.parse_args()


with open(args.i_path) as rf:
    lines = rf.read().splitlines()
    
with open(args.o_path, 'w') as wf:
    wf.write('\n'.join(lines))
# print('\n'.join(lines))