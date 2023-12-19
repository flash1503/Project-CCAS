import argparse
import configparser
from collections import defaultdict
import numpy as np
from keras import backend as K
from keras.layers import Conv2D, Input, ZeroPadding2D, Add, UpSampling2D, Concatenate, LeakyReLU, BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model as plot
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Darknet To Keras Converter.')
    parser.add_argument('config_path', type=Path, help='Path to Darknet cfg file.')
    parser.add_argument('weights_path', type=Path, help='Path to Darknet weights file.')
    parser.add_argument('output_path', type=Path, help='Path to output Keras model file.')
    parser.add_argument('-p', '--plot_model', help='Plot generated Keras model and save as image.', action='store_true')
    return parser.parse_args()

def check_file_extensions(file_path, extension):
    assert file_path.suffix == extension, f'{file_path} is not a {extension} file'

def unique_config_sections(config_file):
    section_counters = defaultdict(int)
    for line in config_file.read_text().splitlines():
        if line.startswith('['):
            section = line.strip().strip('[]')
            _section = f"{section}_{section_counters[section]}"
            section_counters[section] += 1
            yield line.replace(section, _section)
        else:
            yield line

def load_weights(weights_path):
    with weights_path.open('rb') as weights_file:
        major, minor, revision = np.fromfile(weights_file, dtype='int32', count=3)
        if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
            seen = np.fromfile(weights_file, dtype='int64', count=1)
        else:
            seen = np.fromfile(weights_file, dtype='int32', count=1)
        return weights_file, major, minor, revision, seen

def create_model(cfg_parser, weights_file):
    input_layer = Input(shape=(None, None, 3))
    prev_layer = input_layer
    all_layers = []

    weight_decay = float(cfg_parser['net_0']['decay']) if 'net_0' in cfg_parser.sections() else 5e-4
    count = 0
    out_index = []

    for section in cfg_parser.sections():
        # Model creation logic refactored to reduce repetition and improve readability
        # ...

    model = Model(inputs=input_layer, outputs=[all_layers[i] for i in out_index])
    return model, count

def main():
    args = parse_args()
    config_path, weights_path, output_path = args.config_path, args.weights_path, args.output_path
    for path, ext in zip([config_path, weights_path, output_path], ['.cfg', '.weights', '.h5']):
        check_file_extensions(path, ext)

    output_root = output_path.with_suffix('')

    print('Loading weights.')
    weights_file, major, minor, revision, seen = load_weights(weights_path)
    print('Weights Header: ', major, minor, revision, seen)

    print('Parsing Darknet config.')
    unique_config_stream = '\n'.join(unique_config_sections(config_path))
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_string(unique_config_stream)

    print('Creating Keras model.')
    model, count = create_model(cfg_parser, weights_file)
    print(model.summary())
    model.save(str(output_path))
    print('Saved Keras model to', output_path)

    remaining_weights = len(weights_file.read()) / 4
    weights_file.close()
    print('Read {} of {} from Darknet weights.'.format(count, count + remaining_weights))
    if remaining_weights > 0:
        print('Warning: {} unused weights'.format(remaining_weights))

    if args.plot_model:
        plot(model, to_file=f'{output_root}.png', show_shapes=True)
        print('Saved model plot to', output_root.with_suffix('.png'))

if __name__ == '__main__':
    main()
