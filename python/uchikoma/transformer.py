from .circom import *
import numpy as np
from . import model, utils

def register_op_map(op_name):
    def wrapper_func(f):
        def _wrapper(sym: model.Symbol):
            if sym.op != op_name:
                return {}
            return {sym.op: f(sym)}
        return _wrapper
    return wrapper_func

def map_binary_op(sym: model.Symbol, name) -> str:
    A_shape = list(sym.inputs[0].attrs["shape"])
    B_shape = list(sym.inputs[1].attrs["shape"])
    if A_shape == B_shape:
        return "Element{}D{}".format(len(A_shape), name)

    assert len(A_shape) == len(B_shape)
    max_dim = max(len(A_shape), len(B_shape))
    #  A_shape = [1]*max_dim + A_shape
    #  B_shape = [1]*max_dim + B_shape
    equal_dims = []
    for i, (sa, sb) in enumerate(zip(A_shape[-max_dim:], B_shape[-max_dim:])):
        if sa == sb and sa != 1:
            equal_dims.append(i)
        else:
            assert any([sa == 1, sb == 1])
    assert len(equal_dims) == 1, "{}: {} vs. {}".format(
            equal_dims, A_shape, B_shape)
    return "Broadcast{}DAxis{}{}".format(
            max_dim, equal_dims[0], name)

@register_op_map("subtract")
def map_subtract(sym: model.Symbol):
    return map_binary_op(sym, "Sub")

@register_op_map("add")
def map_add(sym: model.Symbol):
    return map_binary_op(sym, "Add")

def map_component(sym: model.Symbol) -> CircomGenerator:
    inputs = sym.inputs
    comp_map = {
        "null": "Input",

        "nn.pad_scalar": "Pad2D",
        "nn.conv2d": "Conv2D_CHW",
        "nn.dense": "Dense2" if len(inputs) == 2 else "Dense",

        **map_subtract(sym),
        **map_add(sym),
        #  "add": "ElementAdd",

        "mul_scalar": "MulScalar",
        "add_scalar": "AddScalar",
        "subtract_scalar": "SubScalar",

        "right_shift": "RightShift",
        "clip": "Clip",

        "image.resize2d": "Resize2D",

        "reshape": "ReShape{}D".format(len(sym.attrs["shape"])),
        #  "reshape": "ReShape" + str(len(sym.attrs["shape"])) + "D",
        "flatten": "Flatten{}D".format(
            len(inputs[0].attrs["shape"]) if inputs else 0),
    }
    return components[comp_map[sym.op]]

#  def shape_adapter(generator: CircomGenerator):
#      """ Change model to adapt circom shape rules by reshape & flatten. """
#      def prod(shape):
#          total = 1
#          for s in shape:
#              total *= s
#          return total

#      count = 0
#      inputs = []
#      for idx, inp in enumerate(generator.inputs):
#          expected_dim = generator.comp.input_dims[idx]
#          if expected_dim == len(inp.shape):
#              pass
#          elif expected_dim == 1:
#              flatten_shape = prod(inp.shape)
#              comp = components["Flatten" + str(len(inp.shape)) + "D"]
#              inp = comp("flatten_" + str(count),
#                      [ inp, ], { "shape": ( flatten_shape, )})
#              count += 1
#              generator
#          else:
#              print("cannot automatically adapt shape")
#              raise RuntimeError(generator.info())

#          inputs.append(inp)


def model2circom(symbol, params):
    generator_map: typing.Dict[str, CircomGenerator] = {}
    circom_ops = set()
    def sym2circom(sym: model.Symbol):
        name = sym.name
        if name in generator_map:
            return

        inputs = [generator_map[i.name] for i in sym.inputs]

        gen = map_component(sym)(name, inputs, sym.attrs)
        circom_ops.add(gen.comp.op_name)
        #  comp.fill_circom()
        generator_map[name] = gen

    model.visit(symbol, sym2circom)
    #  print("Invoked Circoms: \n", "\n".join(circom_ops))

    out = components["Output"](
            "out", [generator_map[symbol.name]],
            { "shape": generator_map[symbol.name].shape })
    return out

def input_json(
        symbol: model.Symbol,
        params: typing.Dict[str, np.ndarray]):
    """ ndarray of str in json format, instead of int """
    def _as_str_data(data):
        if isinstance(data, list):
            return [_as_str_data(d) for d in data]
        assert isinstance(data, int)
        return str(data)

    return {k: _as_str_data(v.tolist()) \
            for k, v in params.items()}


