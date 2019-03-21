"""

"""

import tensorflow as tf
import argparse
from tensorflow.contrib.training.python.training import hparam

from ludwig import LudwigModel

model_definition = {"input_features": [{"name": "doc_text", "type": "text"}],
                    "output_features": [{"name": "class", "type": "category"}]}

ludwig_model = LudwigModel(model_definition)
train_stats = ludwig_model.train(data_csv="gs://skyl-dev-ml/playground/ludwig/train.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train-files',
        nargs='+',
        help='Training file local or GCS',
        default='gs://skyl-dev-ml/aashishdahiya/flowers_aashishdahiya_testV3_20181205_134432/preproc/train*'
    )

    parser.add_argument(
        '--eval-files',
        nargs='+',
        help='Evaluation file local or GCS',
        default='gs://skyl-dev-ml/aashishdahiya/flowers_aashishdahiya_testV3_20181205_134432/preproc/eval*')

    parser.add_argument(
        '--job-dir',
        type=str,
        help='GCS or local dir to write checkpoints and export model',
        default='gs://skyl-dev-ml/aashishdahiya/flowers_aashishdahiya_testV3_20181205_134432')

    parser.add_argument(
        '--num-epochs',
        type=int,
        default=5,
        help='Maximum number of epochs on which to train')

    parser.add_argument(
        '--checkpoint-epochs',
        type=int,
        default=1,
        help='Checkpoint per n training epochs')

    args, _ = parser.parse_known_args()

    hparams = hparam.HParams(**args.__dict__)
    print(hparams)
    # train_and_evaluate(hparams)
