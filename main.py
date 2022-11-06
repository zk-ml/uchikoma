import sys
from os import path

__ROOT__ = path.dirname(path.realpath(__file__))
sys.path.insert(0, path.join(__ROOT__, "python"))

import argparse

parser = argparse.ArgumentParser(
        prog="zkml",
        description="zkml translator from model into circuits",
        )

parser.add_argument("symbol", help="model symbol file name")
# parser.add_argument("ir", help="model ir file name")
parser.add_argument("params", help="model params file name")
parser.add_argument("--info", action="store_true",
        help="print model network graph info")
parser.add_argument("--infer",
        action="store_true",
        help="model inference with random input")
parser.add_argument("--data", metavar="D",
        help="model input data, random if not specified.")
parser.add_argument("--data-dict", metavar="D",
        help="model input dict data, like { 'a': 3 }.")
parser.add_argument("-in", "--input-name",
        metavar="IN",
        help="input name in model, like `input`")
parser.add_argument("-on", "--output-name",
        metavar="ON",
        help="output name in model, like `output`")
parser.add_argument("-o", "--output",
        default="./model", metavar="PATH",
        help="path for generated circom code")

import typing
import numpy as np

import json

from zkml import model, circom, transformer
from zkml import inference as infer
from zkml import circom_impl

def zkml_main():
    args = parser.parse_args()
    print(args)

    np.random.seed(0)

    #  dump_path = "./model/ir_parse"
    #  symbol, params = model.load(dump_path)
    symbol, params = model.ir_load(args.symbol, args.params)
    symbol = model.config(symbol,
            args.input_name, args.output_name)

    #  model.simplify_print(symbol)
    #  print("======= fuse constant =======")

    #  model.simplify_print(symbol)
    symbol, params = infer.fuse_constant(symbol, params)
    symbol = model.fuse_fixed_point_multiply(symbol)
    symbol = model.fuse_scalar_op(symbol, params)
    #  model.simplify_print(symbol)

    if args.info:
        model.simple_raw_print(symbol, params)
        return

    symbol, params = model.fuse_tanh(symbol, params)
    symbol = model.fuse_cast(symbol)
    symbol, params = model.resize_batch(symbol, params)
    symbol = model.shape_adapter(symbol)
    symbol, params = infer.fuse_constant(symbol, params)
    symbol = model.validate_scalar(symbol)
    symbol, params = model.check_params(symbol, params)

    # change into valid symbol name
    new_params = {}
    def _change_name(sym: model.Symbol):
        if model.is_operator(sym, params):
            name = sym.name.replace("%", "O_")
        elif model.is_param(sym, params):
            name = sym.name.replace("%", "P_")
            name = name.replace(".", "_")
            new_params[name] = params[sym.name]
        else:
            name = sym.name.replace("%", "I_")

        return sym.clone(name=name)
    symbol = model.visit(symbol, _change_name)
    params = new_params

    # set input as params
    data = np.array(eval(args.data)) if args.data else None
    data_dict = {k: np.array(v) \
            for v in eval(args.data_dict or "{}").items()}
    data_dict["native_input"] = data
    def _set_input(sym: model.Symbol):
        if model.is_input(sym, params):
            data = data_dict.get(sym.name,
                    data_dict["native_input"])

            shape = sym.attrs["shape"]
            dtype = sym.attrs["dtype"]
            assert "int" in dtype
            if data is None:
                data = np.random.randint(0, 255,
                        size=shape, dtype=dtype)
            assert list(shape) == list(data.shape), (
                    "{}@{} vs. {}").format(sym.name, shape, data.shape)

            print("INPUT {}@{}: {}".format(
                sym.name, data.shape, data.tolist()))
            params[sym.name] = data
    symbol = model.visit(symbol, _set_input)

    if args.infer:
        out = infer.execute(symbol, params)
        def _print(sym: model.Symbol):
            assert sym.name in params
            shape = params[sym.name].shape
            param = params[sym.name].flatten().tolist()
            print("{:20} = {:>30}({:20}) | [{}, ..., {}]".format(
                "{}@({})".format(sym.name[:10],
                    ",".join([str(s) for s in shape])),
                sym.op,
                ", ".join([i.name[:10] for i in sym.inputs]),
                ", ".join([str(p) for p in param[:3]]),
                ", ".join([str(p) for p in param[-3:]]),
                ))
        model.visit(symbol, _print)
        return

    model.simple_raw_print(symbol, params)
    #  model.info(symbol, params)

    # register circom operators
    circom_dir = path.join(__ROOT__, "circuits")
    circom.dir_parse(circom_dir, skips=[
        "util.circom", "Arithmetic.circom",
        "tests", "circomlib-matrix", "circomlib"])
    #  circom.info()

    print(">>> Generating circom code ...")
    out = transformer.model2circom(symbol, params)
    code = circom.generate(out)
    input_json = transformer.input_json(symbol, params)

    print(">>> Generated, dump to {} ...".format(args.output))
    #  print(code)
    with open(args.output + ".circom", "w") as f:
        f.write(code)
    with open(args.output + ".json", "w") as f:
        f.write(json.dumps(input_json, indent=2))

    return

    symbol, params = model.ir_load(
            args.symbol, args.params,
            input_name = "%4",
            output_name = "%186")
    model.dump(symbol, params, dump_path)

    # test model loader
    symbol, params = model.load(dump_path)
    model.info(symbol, params)



if __name__ == "__main__":
    zkml_main()
