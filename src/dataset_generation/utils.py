import json
from typing import Tuple
from generator import Generator
from dclasses import GeneratorDetails

def create_generator(path)->Tuple[Generator, GeneratorDetails]:
    with open(path) as f:
        d = json.load(f)
    param = GeneratorDetails(**d)
    gen = Generator(param)
    return gen, param