#!/bin/bash

# TODO - run your inference Python3 code
# python p2_test.py ./hw3_data/p2_data/images/val ./p2_pred.json ./hw3_data/p2_data/decoder_model.bin
python3 p2_test_beam.py $1 $2 $3

