""" Sample TensorFlow XML-to-TFRecord converter

usage: generate_tfrecord.py [-h] [-x XML_DIR] [-l LABELS_PATH] [-o OUTPUT_PATH] [-i IMAGE_DIR] [-c CSV_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -x XML_DIR, --xml_dir XML_DIR
                        Path to the folder where the input .xml files are stored.
  -l LABELS_PATH, --labels_path LABELS_PATH
                        Path to the labels (.pbtxt) file.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path of output TFRecord (.record) file.
  -i IMAGE_DIR, --image_dir IMAGE_DIR
                        Path to the folder where the input image files are stored. Defaults to the same directory as XML_DIR.
  -c CSV_PATH, --csv_path CSV_PATH
                        Path of output .csv file. If none provided, then no file will be written.
"""

import os
import sys
from pathlib import Path
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse
import re


def _warn_if_not_using_repo_venv():
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    expected_venv = repo_root / '.venv' / 'Scripts' / 'python.exe'
    exe = Path(sys.executable).resolve()

    if expected_venv.exists() and exe != expected_venv.resolve():
        print(
            "[generate_tfrecord] Warning: you are not running the repo venv.\n"
            f"  Current python: {exe}\n"
            f"  Expected venv:  {expected_venv}\n\n"
            "This commonly causes TensorFlow/protobuf import errors (e.g. missing google.protobuf.runtime_version).\n"
            "Run this script with:\n"
            f"  {expected_venv} Tensorflow\\scripts\\generate_tfrecord.py ...\n",
            file=sys.stderr,
        )

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)

_warn_if_not_using_repo_venv()

try:
    import tensorflow.compat.v1 as tf
except ImportError as e:
    message = str(e)
    if "runtime_version" in message and "google.protobuf" in message:
        raise ImportError(
            message
            + "\n\nLikely cause: TensorFlow is importing against an incompatible protobuf in this interpreter. "
            + "Use the repo venv (.venv) or upgrade protobuf in the same interpreter you run this script with."
        ) from e
    raise
from PIL import Image
from collections import namedtuple

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TensorFlow XML-to-TFRecord converter")
parser.add_argument("-x",
                    "--xml_dir",
                    help="Path to the folder where the input .xml files are stored.",
                    type=str)
parser.add_argument("-l",
                    "--labels_path",
                    help="Path to the labels (.pbtxt) file.", type=str)
parser.add_argument("-o",
                    "--output_path",
                    help="Path of output TFRecord (.record) file.", type=str)
parser.add_argument("-i",
                    "--image_dir",
                    help="Path to the folder where the input image files are stored. "
                         "Defaults to the same directory as XML_DIR.",
                    type=str, default=None)
parser.add_argument("-c",
                    "--csv_path",
                    help="Path of output .csv file. If none provided, then no file will be "
                         "written.",
                    type=str, default=None)

args = parser.parse_args()

if args.image_dir is None:
    args.image_dir = args.xml_dir


def _bytes_feature(value: bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list(values)))


def _int64_feature(value: int):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _int64_list_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v) for v in values]))


def _float_list_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v) for v in values]))


def _load_label_map_dict(pbtxt_path: str):
    """Minimal parser for TF Object Detection API style label_map.pbtxt.

    Expected blocks like:
      item { id: 1 name: 'hello' }
    """

    with open(pbtxt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    items = re.findall(r'item\s*\{(.*?)\}', text, flags=re.DOTALL)
    label_map_dict = {}
    for item in items:
        id_match = re.search(r'\bid\s*:\s*(\d+)', item)
        name_match = re.search(r"\bname\s*:\s*'([^']+)'", item) or re.search(
            r'\bname\s*:\s*"([^"]+)"', item
        )
        if not id_match or not name_match:
            continue
        label_map_dict[name_match.group(1)] = int(id_match.group(1))

    if not label_map_dict:
        raise ValueError(f"No labels parsed from: {pbtxt_path}")
    return label_map_dict


label_map_dict = _load_label_map_dict(args.labels_path)


def _normalize_label(name: str) -> str:
    # Normalize common variations: casing and whitespace.
    return ''.join(str(name).split()).lower()


_label_map_norm = {_normalize_label(k): v for k, v in label_map_dict.items()}


def _append_label_to_pbtxt(pbtxt_path: str, name: str, new_id: int) -> None:
    p = Path(pbtxt_path)
    existing = p.read_text(encoding='utf-8')
    addition = f"\nitem {{ \n\tname:'{name}'\n\tid:{new_id}\n}}\n"
    p.write_text(existing.rstrip() + addition + "\n", encoding='utf-8')


def _ensure_label_id(row_label: str) -> int:
    """Return the label id for row_label.

    If missing, automatically appends it to the label map pbtxt with a new id,
    updates in-memory maps, and continues.
    """

    # Exact match
    if row_label in label_map_dict:
        return label_map_dict[row_label]

    # Normalized match
    key = _normalize_label(row_label)
    if key in _label_map_norm:
        return _label_map_norm[key]

    # Auto-add (stable fix for labels like 'ww' that keep disappearing)
    next_id = (max(label_map_dict.values()) if label_map_dict else 0) + 1
    _append_label_to_pbtxt(args.labels_path, row_label, next_id)
    label_map_dict[row_label] = next_id
    _label_map_norm[_normalize_label(row_label)] = next_id
    print(
        f"[generate_tfrecord] Added missing label '{row_label}' to {args.labels_path} with id {next_id}.",
        file=sys.stderr,
    )
    return next_id


def xml_to_csv(path):
    """Iterates through all .xml files (generated by labelImg) in a given directory and combines
    them in a single Pandas dataframe.

    Parameters:
    ----------
    path : str
        The path containing the .xml files
    Returns
    -------
    Pandas DataFrame
        The produced dataframe
    """

    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def class_text_to_int(row_label):
    return _ensure_label_id(row_label)


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    image_path = os.path.join(path, '{}'.format(group.filename))
    if not tf.io.gfile.exists(image_path):
        return None

    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/filename': _bytes_feature(filename),
        'image/source_id': _bytes_feature(filename),
        'image/encoded': _bytes_feature(encoded_jpg),
        'image/format': _bytes_feature(image_format),
        'image/object/bbox/xmin': _float_list_feature(xmins),
        'image/object/bbox/xmax': _float_list_feature(xmaxs),
        'image/object/bbox/ymin': _float_list_feature(ymins),
        'image/object/bbox/ymax': _float_list_feature(ymaxs),
        'image/object/class/text': _bytes_list_feature(classes_text),
        'image/object/class/label': _int64_list_feature(classes),
    }))
    return tf_example


def main(_):

    writer = tf.io.TFRecordWriter(args.output_path)
    path = os.path.join(args.image_dir)
    examples = xml_to_csv(args.xml_dir)
    grouped = split(examples, 'filename')
    skipped = 0
    for group in grouped:
        tf_example = create_tf_example(group, path)
        if tf_example is None:
            skipped += 1
            continue
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecord file: {}'.format(args.output_path))
    if skipped:
        print(f'Skipped {skipped} examples due to missing image files.')
    if args.csv_path is not None:
        examples.to_csv(args.csv_path, index=None)
        print('Successfully created the CSV file: {}'.format(args.csv_path))


if __name__ == '__main__':
    main(None)
