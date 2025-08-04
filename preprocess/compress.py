import os
import argparse
from PIL import Image


def parser_args():
    parser = argparse.ArgumentParser(description='compress parameters')
    parser.add_argument('--set', type=str, default='train')
    parser.add_argument('--class_dir', type=str, default='0_N')
    parser.add_argument('--quality', type=int, default=95)
    return parser.parse_args()


def main():
    args = parser_args()
    data_dir = '../datasets/BRACS/BRACS_RoI/latest_version/{}/{}'.format(args.set, args.class_dir)
    compress_dir = '../datasets/BRACS/BRACS_RoI/compressed/{}/{}'.format(args.set, args.class_dir)
    if not os.path.exists(compress_dir):
        os.makedirs(compress_dir)

    for img_name in os.listdir(data_dir):
        image_path = os.path.join(data_dir, img_name)
        compress_path = os.path.join(compress_dir, img_name)
        size = os.path.getsize(image_path)

        i = 0
        while size > 20 * 1e6 and i < 20:
            print(image_path, size / 1e6, i)
            image = Image.open(image_path)

            if i > 0:
                img_name_jpg = img_name.replace(".png", ".jpg")
                compress_path = os.path.join(compress_dir, img_name_jpg)
                image.save(compress_path, quality=100 - 5 * i)
            else:
                image.save(compress_path, quality=95)

            size = os.path.getsize(compress_path)
            i = i + 1

        if i == 20:
            print('Fail:', compress_path)

    print("Done!")


if __name__ == '__main__':
    main()
