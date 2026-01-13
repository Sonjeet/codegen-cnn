#!/usr/bin/env python3

import subprocess
import os
import argparse

# TODO: should we add a check to see if 3p deps exist locally - if not then brew install?

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
BUILD_DIR = os.path.join(ROOT_DIR, 'build')

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', help='run unit tests after building',
                                 action='store_true')
parser.add_argument('-e', '--exec', help='run the built executable',
                                 action='store_true')
parser.add_argument('-c', '--clean', help='run a clean make',
                                 action='store_true')

args = parser.parse_args()

if not os.path.isdir(BUILD_DIR):
    print(f'Out directory {BUILD_DIR} doesn\'t exist. Creating dir now..')
    os.mkdir(BUILD_DIR)
    print(f'Created dir: {BUILD_DIR}')

# configure cmake to write to build dir
subprocess.run(['cmake', '-B', 'build', '.'])
# compile with build files in the build dir

# TODO: obvs can't clean the build and then run an executable - fix bug
if args.clean:
    subprocess.run(['cmake', '--build', './build', '--target', 'clean'])
else:
    subprocess.run(['cmake', '--build', './build'])

if args.exec:
    print('Running executable')
    subprocess.run(['./cnn'], cwd='build')
if args.test:
    print('Running unit tests')
    subprocess.run(['ctest'], cwd='build/test')