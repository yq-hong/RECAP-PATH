import os
import argparse
from pathlib import Path


def parser_args():
    parser = argparse.ArgumentParser(description='compress parameters')
    parser.add_argument('--set', type=str, default='train')
    # 0_N, 1_PB, 2_UDH, 3_FEA, 4_ADH, 5_DCIS, 6_IC
    parser.add_argument('--class_dir', type=str, default=None)
    return parser.parse_args()


def get_img_paths(data_dir):
    img_names = os.listdir(data_dir)
    img_paths = [os.path.join(data_dir, img_name) for img_name in img_names]

    for i in range(len(img_paths)):
        img_path = img_paths[i]
        size = os.path.getsize(img_path)

        if size > 20 * 1e6:
            img_path_compress = img_path.replace("latest_version", "compressed")
            try:
                size = os.path.getsize(img_path_compress)
                if size > 20 * 1e6:
                    img_path_compress = img_path_compress.replace(".png", ".jpg")
            except:
                img_path_compress = img_path_compress.replace(".png", ".jpg")

            size = os.path.getsize(img_path_compress)
            if size > 20 * 1e6:
                print('Error image name!')

            img_paths[i] = img_path_compress

    return img_paths


def main():
    args = parser_args()
    if args.class_dir != None:
        data_dir = '../datasets/BRACS/BRACS_RoI/latest_version/{}/{}'.format(args.set, args.class_dir)
        img_paths = get_img_paths(data_dir)

        # file_path = Path('file_names/{}_{}.txt'.format(args.set, args.class_dir))
        # file_path.write_text('\n'.join(img_paths) + '\n')
        file_path = 'file_names/{}_{}.txt'.format(args.set, args.class_dir)
        with open(file_path, 'w') as file:
            for element in img_paths:
                file.write(f"{element}\n")

        print(f"List saved to {file_path}")

    else:
        data_dirs_BT = ['../datasets/BRACS/BRACS_RoI/latest_version/{}/0_N'.format(args.set),
                        '../datasets/BRACS/BRACS_RoI/latest_version/{}/1_PB'.format(args.set),
                        '../datasets/BRACS/BRACS_RoI/latest_version/{}/2_UDH'.format(args.set)]
        data_dirs_MT = ['../datasets/BRACS/BRACS_RoI/latest_version/{}/5_DCIS'.format(args.set),
                        '../datasets/BRACS/BRACS_RoI/latest_version/{}/6_IC'.format(args.set)]
        data_dirs_AT = ['../datasets/BRACS/BRACS_RoI/latest_version/{}/3_FEA'.format(args.set),
                        '../datasets/BRACS/BRACS_RoI/latest_version/{}/4_ADH'.format(args.set)]

        img_paths = []
        for data_dir in data_dirs_BT:
            img_paths += get_img_paths(data_dir)
        file_path = '../file_names/{}_BT.txt'.format(args.set)
        with open(file_path, 'w') as file:
            for element in img_paths:
                file.write(f"{element}\n")
        print(f"List saved to {file_path}")

        img_paths = []
        for data_dir in data_dirs_MT:
            img_paths += get_img_paths(data_dir)
        file_path = '../file_names/{}_MT.txt'.format(args.set)
        with open(file_path, 'w') as file:
            for element in img_paths:
                file.write(f"{element}\n")
        print(f"List saved to {file_path}")

        img_paths = []
        for data_dir in data_dirs_AT:
            img_paths += get_img_paths(data_dir)
        file_path = '../file_names/{}_AT.txt'.format(args.set)
        with open(file_path, 'w') as file:
            for element in img_paths:
                file.write(f"{element}\n")
        print(f"List saved to {file_path}")

    print("Done!")


if __name__ == '__main__':
    main()
