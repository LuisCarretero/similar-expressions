import json
from typing import Tuple
from custom.generator import Generator
from custom.dclasses import GeneratorDetails

def create_generator(path)->Tuple[Generator, GeneratorDetails]:
    with open(path) as f:
        d = json.load(f)
    param = GeneratorDetails(**d)
    gen = Generator(param)
    return gen, param