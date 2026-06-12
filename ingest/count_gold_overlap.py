import os
import glob
import argparse


def count_overlap(graphml_dir, p_dir):
    graphml_stems = {
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(graphml_dir, '*.graphml'))
    }
    p_stems = {
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(p_dir, '*.p'))
    }

    overlap = graphml_stems & p_stems

    print(f'graphml files : {len(graphml_stems):,}')
    print(f'.p files      : {len(p_stems):,}')
    print(f'overlap       : {len(overlap):,}')
    return overlap


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Count sentences that have both a .graphml lattice and a .p gold annotation.')
    parser.add_argument('graphml_dir', help='Directory containing .graphml files')
    parser.add_argument('p_dir', help='Directory containing .p gold files')
    args = parser.parse_args()

    count_overlap(args.graphml_dir, args.p_dir)
