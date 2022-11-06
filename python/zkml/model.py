from __future__ import annotations

import json
import math
import typing
from functools import wraps
from dataclasses import dataclass, is_dataclass, fields
import re

import tvm
import numpy as np
from tvm import relay, IRModule as ir

from . import utils

@dataclass
class Symbol:
    op: str
    name: str
    inputs: typing.Dict[Symbol]
    inputs_raw: typing.Dict[typing.List[int]]
    attrs: typing.Dict[str, str]

    @classmethod
    def variable(cls, name, **attrs):
        return cls("null", name, [], [], attrs)

    def find(self, child_name) -> typing.Optional[Symbol]:
        sym_list: typing.List[Symbol] = []
        _topo_sort(self, sym_list)
        for sym in sym_list:
            if sym.name == child_name:
                return sym
        return None

    def as_parameter(self):
        self.op = "null"
        self.inputs = []
        self.inputs_raw = []
        assert "dtype" in self.attrs, self.info()
        self.attrs = {
            "shape": self.attrs["shape"],
            "dtype": self.attrs["dtype"],
        }
        return self

    def clone(self, **kw) -> Symbol:
        data = dict((f.name, getattr(self, f.name)) \
                for f in fields(self))
        data.update(kw)
        return Symbol(**data)

    def info(self):
        inputs_info = [
            "{}@{}".format(i.name, i.attrs.get("shape", None)) \
            for i in self.inputs ]
        return "{} = {}({}) /* attrs */ \t{}".format(
            self.name, self.op,
            ", ".join(inputs_info),
            self.attrs)

def _topo_sort(symbol, sym_list: typing.List[Symbol]):
    if sym_list.count(symbol) > 0:
        return
    for c in symbol.inputs:
        _topo_sort(c, sym_list)
    sym_list.append(symbol)

def visit(symbol, callback) -> Symbol:
    sym_list: typing.List[Symbol] = []
    _topo_sort(symbol, sym_list)

    sym_map = {}
    for sym in sym_list:
        inputs = [sym_map[c.name] for c in sym.inputs]
        sym = sym.clone(inputs=inputs)
        # pre-clone symbol, to avoid misleading usage in callback
        out = callback(sym.clone()) or sym
        assert isinstance(out, Symbol)
        sym_map[sym.name] = out
    return sym_map[symbol.name]


#  def _visit_impl(symbol, callback, visit_map):
#      if symbol.name in visit_map:
#          return visit_map[symbol.name]

#      visit_map[symbol.name] = None
#      child_exec = lambda child: _visit_impl(
#          child, callback, visit_map)

#      inputs = [child_exec(c) for c in symbol.inputs]
#      symbol = symbol.clone(inputs=inputs)

#      sym: Symbol = callback(symbol) or symbol
#      assert isinstance(sym, Symbol)
#      visit_map[symbol.name] = sym

#      return sym

#  CallbackType = typing.Callable[(Symbol), typing.Optional[Symbol]]

#  def visit(
#          symbol: Symbol,
#          callback: CallbackType) -> Symbol:
#      visit_map = {}
#      return _visit_impl(symbol, callback,
#              visit_map=visit_map)

def transform(f: CallbackType):
    @wraps(f)
    def _wrapper_visit(symbol):
        return visit(symbol, f)
    return _wrapper_visit

def simple_raw_print(symbol, params={}):
    info = { "op": 0, "param": 0 }
    def _simple_visit(sym):
        if not is_operator(sym):
            print("{:68} /* attrs */ \t{}".format(
                sym.name, sym.attrs))
            if is_param(sym, params):
                info["param"] += utils.product(sym.attrs["shape"])
            return

        info["op"] += 1
        print("{:15} = {:>20}{:30} /* attrs */ \t{}".format(
            sym.name, sym.op,
            "(" + ", ".join([i.name for i in sym.inputs]) + ")",
            sym.attrs,
        ))
    visit(symbol, _simple_visit)
    print("="*50)
    print("Operators: {} | Parameters: {}".format(
        info["op"], info["param"]))
    print("="*50)



def dump(symbol, params, dump_path):
    dict_nodes = []
    def _to_dict(sym):
        dict_nodes.append({
            "op": sym.op,
            "name": sym.name,
            "inputs": [i.name for i in sym.inputs],
            "attrs": sym.attrs,
        })
    visit(symbol, _to_dict)

    dump_json = {
        "nodes": dict_nodes,
        "head": symbol.name,
    }
    with open(dump_path + ".json", "w") as f:
        json.dump(dump_json, f, indent=2)

    np.save(dump_path + ".params", params)

def load(model_path):
    with open(model_path + ".json", "r") as f:
        model_json = json.load(f)

    # assume the node order is topological.
    symbol_map = {}
    for node in model_json["nodes"]:
        inputs = [symbol_map[i] for i in node["inputs"]]
        sym = Symbol(
                node["op"], node["name"],
                inputs, node["inputs"],
                node["attrs"])
        symbol_map[node["name"]] = sym

    symbol = symbol_map[model_json["head"]]

    params = np.load(
            model_path + ".params.npy",
            allow_pickle=True)
    params = params.item()
    return symbol, params

def info(symbol, params):
    inputs = []
    param_syms = []
    outputs = []
    operators = []

    total_operators = 0
    operator_info = {}

    useful_params = {}
    def _collect_info(sym):
        if is_operator(sym, params):
            operator_info.setdefault(sym.op, 0)
            operator_info[sym.op] += 1
            total_operators = 0
            operators.append(sym)
        elif is_input(sym, params):
            inputs.append(sym)
        elif is_param(sym, params):
            useful_params[sym.name] = params[sym.name]
            param_syms.append(sym)

    visit(symbol, _collect_info)

    print()
    print("======= Model Infomation ======")
    print("Inputs: {}\t Params: {}\t Operator: {}".format(
        len(inputs), len(useful_params.keys()),
        len(operator_info.keys())
    ))

    print(">>        Input               <")
    for inp in inputs:
        print("{} attrs={}".format(inp.name, inp.attrs))
    #  print(">>        Params              <")
    #  for k, p in useful_params.items():
    #      print("{:20} shape={}".format(k, p.shape))
    print(">>        ParamSymbols        <")
    for p in param_syms:
        print("{:20} attrs={}".format(p.name, p.attrs))
    print(">>        Operators           <")
    #  for op in operators:
    #      print("{:20} attrs={}".format(op.op, op.attrs))
    for op, count in operator_info.items():
        print("{:25} count={}".format(op, count))



def is_operator(symbol: Symbol, params = {}):
    return symbol.op != "null"

def is_input(symbol: Symbol, params):
    return symbol.op == "null" and symbol.name not in params

def is_param(symbol: Symbol, params):
    return symbol.op == "null" and symbol.name in params

def json_load(symbol_path, ir_path, params_path):
    # mod = relay.build_module.load_module(symbol_path)
    # print(mod)
    # relay.build_module.build

    with open(symbol_path, "r") as f:
        sym_json = json.load(f)
    nodes = sym_json["nodes"]

    symbol_map: typing.Dict[int, Symbol] = {}
    for idx, node in enumerate(nodes):
        sym = Symbol(
                node["op"], node["name"],
                [], node["inputs"], {})

        for inp_raw in sym.inputs_raw:
            assert len(inp_raw) == 3
            assert inp_raw[0] in symbol_map
            assert inp_raw[1] == 0
            assert inp_raw[2] == 0
            sym.inputs.append(symbol_map[inp_raw[0]])

        sym.attrs = node.get("attrs", {})
        if is_operator(sym): # parse func name
            sym.op = sym.attrs["func_name"]


        symbol_map[sym.name] = sym

    heads = sym_json["heads"]
    symbol = []
    for head in heads:
        assert len(head) == 3
        assert head[0] in symbol_map
        assert head[1] == 0
        assert head[2] == 0
        symbol.append(symbol_map[head[0]])

    assert len(symbol) == 1
    symbol = symbol[0]

    print(symbol.to_string())

    with open(params_path, "rb") as f:
        params: typing.Dict[str, tvm.runtime.ndarray.NDArray] = tvm.runtime.load_param_dict(f.read())

    print(type(params))
    print(type(params["p1"]))
    print(params["p1"])
    print(type(params["p35"]))
    print(params["p33"].shape)

    operator_names = []
    inputs, weights, others = [], [], []
    for node in symbol_map.values():
        if is_operator(node, params):
            operator_names.append(node.name)
        elif is_input(node, params):
            inputs.append(node)
        elif is_param(node, params):
            weights.append(node)
        else:
            print("unrecognized node: {}".format(
                node.to_string(depth=1, with_attrs=True)))
            others.append(node)

    print("inputs: ", [i for i in inputs])
    print("operators: ", len(operator_names))

def ir_load(symbol_path, params_path):
    with open(params_path, "rb") as f:
        params: typing.Dict[str, tvm.runtime.ndarray.NDArray]  = tvm.runtime.load_param_dict(f.read())

    # pad prefix:% to parameters
    params = {"%" + k: v.numpy() for k, v in params.items()}

    with open(symbol_path, "r") as f:
        ir_str = f.read().replace("\n", "")

    # parse metadata
    m = re.search("#\[metadata\]({.*})", ir_str)
    if m is not None:
        meta_str = m.group(1)
        #  print(meta_str)
        metadata: typing.Dict[str, tvm.ir.container.Array] = tvm.ir.load_json(meta_str)
        for k, v in metadata.items():
            constant = []
            #  a = tvm.relay.expr.Constant(tvm.nd.array([3]))
            #  print(repr(a))
            for d in v:
                mret = re.search("Constant\(\[(\d+)\]\)", repr(d))
                constant.append(int(mret.group(1)))
            params[k] = np.array(constant)
            #  print(k, v, type(v[0]), params[k])


    # print(ir_str)
    m = re.search("def @main\((.*)\) -> ([^{]*) {([^}]*)}", ir_str)
    params_info = m.group(1)
    ret = m.group(2)
    main = m.group(3)

    # validate params shape
    # create params symbol (include inputs)
    symbol_map = {}

    params_info = re.findall(
        "(.*?): Tensor\[(\(.*?\)), (.*?)\] /\* .*? \*/,?",
        params_info)
    for info in params_info:
        name = info[0].strip()
        shape = eval(info[1])
        if isinstance(shape, int):
            shape = ( shape, )
        dtype = info[2]
        if name in params:
            assert list(params[name].shape) == list(shape)
            assert params[name].dtype.name == dtype
        symbol_map[name] = Symbol.variable(
                name, shape=shape, dtype=dtype)

    # parse network graph
    output_name = "output"
    statements = main.split(";")
    for idx, statement in enumerate(statements):
        sret = re.search("((.*) = )?([^%]+)\((.*)\) /\*(.*)\*/", statement.strip())
        name = sret.group(2) or output_name
        op_name = sret.group(3)

        anno = sret.group(5)
        anno_ret = re.search("ty=Tensor\[(.*), (.*)\]", anno)
        shape = eval(anno_ret.group(1))
        if isinstance(shape, int):
            shape = (shape,)
        dtype = anno_ret.group(2)

        sym = Symbol(op_name, name,
                [], [], {})

        sym.attrs.update({ "shape": shape, "dtype": dtype, })

        args = sret.group(4).strip()
        args = re.findall("([^,^/]*\[[^=]+\]|[^,^/^[]+)(?: /\*(.*)\*/)?(,|$)", args)
        for a in args:
            arg_name = a[0].strip()
            arg_shape = a[1].strip()

            if "=" in arg_name: # attribute
                attr_name, attr_value = arg_name.split("=")
                try:
                    if attr_value.endswith("f"):
                        attr_value = attr_value[:-1]
                    attr_value = eval(attr_value)
                except Exception:
                    pass
                sym.attrs[attr_name] = attr_value
            else: # starts with % or constant, such as 64f
                if not arg_name.startswith("%"):
                    if "relay.Constant" in arg_name:
                        iret = re.search("relay.Constant\]\[(\d+)\]", arg_name)
                        index = int(iret.group(1))
                        value = params["relay.Constant"][index]
                        #  print(arg_name, " index:[", index, "] value is :", value)
                    else:
                        if arg_name.endswith("f"):
                            arg_name = arg_name[:-1]
                        value = eval(arg_name)

                    # create params if not exist or override
                    arg_name = "const_" + str(value)
                    params[arg_name] = np.array(value)
                    symbol_map[arg_name] = Symbol.variable(
                            arg_name, shape="()")
                elif arg_name not in symbol_map:
                    assert False
                    #  symbol_map[arg_name] = Symbol.variable(
                    #          arg_name, shape=inputs_info[arg_name])
                sym.inputs.append(symbol_map[arg_name])

        assert sym.name not in symbol_map, (
                "duplicated symbol: {}").format(sym.name)
        symbol_map[sym.name] = sym

    #  print(symbol_map.keys())
    #  print(symbol_map["%186"])
    #  print(symbol_map["%186"].simple_string())

    return symbol_map[output_name], params


def config(symbol, input_name = None, output_name = None):
    output_name = output_name or symbol.name
    def replace_input(sym: Symbol):
        if sym.name == input_name:
            sym = sym.clone().as_parameter()
        return sym

    symbol = visit(symbol, replace_input)
    out = symbol.find(output_name)
    assert out is not None, (
        "{} not in model, maybe the {}? "
        "--info for more information"
    ).format(output_name, symbol.name)
    return out

    #  #  simplify_print(new_symbol_map[output_name])
    #  #  new_symbol_map[output_name].simple_print()
    #  return new_symbol_map[output_name], new_params

def check_params(symbol, params):
    new_params = {}
    def _check_params(sym):
        if is_param(sym, params):
            new_params[sym.name] = params[sym.name]
    visit(symbol, _check_params)
    return symbol, new_params

#  def np_exec(sym, inputs):
#      op_name = sym.op
#      if op_name == "cast":
#          assert len(inputs) == 1
#          dtype_map = {
#              "int32": "i4",
#          }
#          return np.ndarray.astype(inputs[0],
#                  dtype=dtype_map[sym.attrs["dtype"]])
#      elif op_name == "sum":
#          assert len(inputs) == 1
#          max_axis = len(inputs[0].shape)
#          axes = [a+max_axis if a < 0 else a \
#                  for a in sym.attrs["axis"]]
#          assert all([(a > 0 and a < max_axis) for a in axes])
#          out = inputs[0]
#          for axis in reversed(sorted(axes)):
#              out = np.ndarray.sum(out, axis=axis)
#          #  print(list(reversed(sorted(axes))), out.shape)
#          assert out.shape == sym.attrs["shape"]
#          return out

#          #  assert len(axes) == 1, sym.info()
#          #  return np.ndarray.sum(inputs[0], axis=axes[0])
#      elif op_name == "multiply":
#          assert len(inputs) == 2, sym.info()
#          return inputs[0] * inputs[1]
#      elif op_name == "reshape":
#          assert len(inputs) == 1
#          return inputs[0].reshape(sym.attrs["newshape"])
#      elif op_name == "expand_dims":
#          assert len(inputs) == 1
#          out = inputs[0]
#          for _ in range(sym.attrs["num_newaxis"]):
#              out = np.expand_dims(out, axis=sym.attrs["axis"])
#          assert out.shape == sym.attrs["shape"], sym.info()
#          return out

#      else:
#          raise NotImplementedError(sym.info())

#  def fuse_constant(symbol, params):
#      def _fuse_constant(sym):
#          if not is_operator(sym, params):
#              return
#          if all([is_param(i, params) for i in sym.inputs]):
#              inputs: typing.List[np.ndarray] = \
#                      [params[i.name] for i in sym.inputs]
#              params[sym.name] = np_exec(sym, inputs)
#              sym = sym.clone().as_parameter()
#          return sym
#      return visit(symbol, _fuse_constant), params

@transform
def fuse_cast(sym):
    if sym.op == "cast":
        #  assert "int" in sym.inputs[0].attrs["dtype"], sym.inputs[0].info()
        #  assert "int" in sym.attrs["dtype"], sym.info()
        return sym.inputs[0]

def _copy_attrs(attrs, *keys, **kw):
    return {
        "shape": attrs["shape"],
        "dtype": attrs["dtype"],
        **{k: attrs[k] for k in keys},
        **kw,
    }

@transform
def fuse_fixed_point_multiply(sym: Symbol):
    if sym.op != "fixed_point_multiply":
        return

    attrs = sym.attrs
    out = Symbol("mul_scalar", sym.name + "_mul",
            sym.inputs, [],
            _copy_attrs(attrs, scalar=attrs["multiplier"]))

    shift_bit = sym.attrs["shift"]
    assert shift_bit <= 0
    shift_bit = -shift_bit
    if shift_bit > 0:
        out = Symbol("right_shift", sym.name + "_shift",
            [out], [], _copy_attrs(attrs, shift_bit=shift_bit))
    return out

@transform
def validate_scalar(sym: Symbol):
    scalar = sym.attrs.get("scalar", 0)
    assert isinstance(scalar, int), sym.info()

def fuse_scalar_op(symbol: Symbol, params):
    def _fuse(sym: Symbol):
        if not is_operator(sym):
            return

        scalars = []
        inputs = []
        for inp in sym.inputs:
            # if params' shape is 1, try to cast into scalar.
            if is_param(inp, params) and utils.product(params[inp.name].shape) == 1:
                scalars.append(params[inp.name].item())
            else:
                inputs.append(inp)

        if len(scalars) == 1:
            sym = sym.clone(op=sym.op+"_scalar", inputs=inputs,)
            sym.attrs["scalar"] = scalars[0]
        return sym
    return visit(symbol, _fuse)

def _shape_adapter(sym: Symbol):
    supported_ops = [
        "clip",
        "mul_scalar", "add_scalar", "subtract_scalar",
        "right_shift",
    ]
    if sym.op not in supported_ops:
        inputs = []
        for inp in sym.inputs:
            if "orig_shape" in inp.attrs:
                inp = Symbol("reshape", inp.name + "_r",
                        [ inp, ], [],
                        { "shape": inp.attrs["orig_shape"],
                            "dtype": sym.attrs["dtype"],
                        })
            inputs.append(inp)
        sym = sym.clone(inputs=inputs)
        return sym

    if len(sym.attrs["shape"]) == 1:
        return

    input_shape = list(sym.inputs[0].attrs["shape"])
    orig_shape = list(sym.inputs[0].attrs.get(
        "orig_shape", input_shape))
    shape_one = utils.product(input_shape)

    inputs = []
    for inp in sym.inputs:
        shape = list(inp.attrs["shape"])
        assert input_shape == shape
        if len(shape) != 1:
            inp = Symbol("flatten", sym.inputs[0].name + "_f",
                    [ sym.inputs[0], ], [],
                    { "shape": ( shape_one, ),
                        "dtype": sym.attrs["dtype"], })
        inputs.append(inp)

    #  assert list(sym.attrs["shape"]) == input_shape, sym.info()
    sym = sym.clone(inputs=inputs)
    sym.attrs.update({
        "shape": ( shape_one, ),
        "orig_shape": orig_shape,
        })
    return sym

def shape_adapter(symbol: Symbol):
    symbol = visit(symbol, _shape_adapter)

    if "orig_shape" in symbol.attrs:
        symbol = Symbol("reshape", symbol.name + "_r",
                [ symbol, ], [],
                {
                    "shape": symbol.attrs["orig_shape"],
                    "dtype": symbol.attrs["dtype"],
                })

    def _clean_attrs(sym: Symbol):
        if "orig_shape" in sym.attrs:
            del sym.attrs["orig_shape"]
    symbol = visit(symbol, _clean_attrs)
    return symbol



def resize_batch(symbol, params, batch_size=1):
    # temporary set batch size, hook!!!
    def _change_batch_size(sym: model.Symbol):
        if is_param(sym, params):
            return

        if sym.op in ["subtract"]:
            inputs = []
            for inp in sym.inputs:
                shape = inp.attrs["shape"]
                if is_param(inp, params) and len(shape) == 4:
                    assert shape[0] == 1
                    param = params[inp.name]
                    inp = inp.clone(name=inp.name+"_dim3")
                    inp.attrs["shape"] = shape[1:]
                    params[inp.name] = param.reshape(shape[1:])
                inputs.append(inp)
            sym = sym.clone(inputs=inputs)

        assert "shape" in sym.attrs, sym.info()
        shape = list(sym.attrs["shape"])
        assert shape[0] == 64
        #  shape[0] = 1
        sym.attrs["shape"] = shape[1:]
    symbol = visit(symbol, _change_batch_size)
    return symbol, params


def fuse_tanh(symbol, params):
    def _tanh(sym: Symbol):
        if sym.op != "tanh":
            return

        mul_input = sym.inputs[0]
        assert mul_input.op == "multiply"

        return sym
    return visit(symbol, _tanh), params


